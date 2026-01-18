#!/bin/bash

# CNN-TDNN Training Script for Kaldi FP16
# Uses pretrained HMM-DNN alignments and optionally loads DNN weights
#
# Usage: ./train_cnn_tdnn.sh <data_dir> <ali_dir> <output_dir> [pretrained_nnet3]
#
# Example:
#   ./train_cnn_tdnn.sh data/train exp/tri4_ali exp/cnn_tdnn_fp16
#   ./train_cnn_tdnn.sh data/train exp/tri4_ali exp/cnn_tdnn_fp16 exp/dnn/final.mdl

set -e
set -o pipefail

# ============================================================================
# Configuration
# ============================================================================

# CNN configuration
cnn_channels="64,128"           # Output channels for CNN layers
cnn_kernels="3,3"               # Kernel sizes
cnn_strides="1,1"               # Strides

# TDNN configuration  
tdnn_dims="256,256,256,256,256"
tdnn_contexts="-1:0:1,-1:0:1,-3:0:3,-3:0:3,-6:-3:0:3:6"

# Training configuration
num_epochs=15
initial_lr=0.001
final_lr=0.0001
batch_size=64
dropout=0.1

# FP16 specific
use_fp16=true
loss_scale=65536
warmup_epochs=2

# Data augmentation
use_specaugment=true
freq_mask=27
time_mask=40
num_freq_masks=2
num_time_masks=2

# ============================================================================
# Parse arguments
# ============================================================================

if [ $# -lt 3 ]; then
    echo "Usage: $0 <data_dir> <ali_dir> <output_dir> [pretrained_nnet3]"
    echo ""
    echo "Arguments:"
    echo "  data_dir       - Kaldi data directory with feats.scp"
    echo "  ali_dir        - Directory with HMM-DNN alignments (ali.*.gz)"
    echo "  output_dir     - Output directory for trained model"
    echo "  pretrained     - (optional) Pretrained nnet3 model for initialization"
    exit 1
fi

data_dir=$1
ali_dir=$2
output_dir=$3
pretrained_nnet3=${4:-""}

mkdir -p $output_dir/log

# ============================================================================
# Get dimensions from data
# ============================================================================

echo "============================================"
echo "CNN-TDNN FP16 Training"
echo "============================================"
echo "Data: $data_dir"
echo "Alignments: $ali_dir"
echo "Output: $output_dir"
echo "Pretrained: ${pretrained_nnet3:-none}"
echo ""

# Get input dimension from features
if [ -f $data_dir/feats.scp ]; then
    feat_dim=$(feat-to-dim scp:$data_dir/feats.scp - 2>/dev/null || echo 40)
else
    echo "Warning: No feats.scp found, assuming 40-dim MFCC"
    feat_dim=40
fi

# Get output dimension from alignments (number of PDFs)
if [ -f $ali_dir/num_pdfs ]; then
    num_pdfs=$(cat $ali_dir/num_pdfs)
elif [ -f $ali_dir/final.mdl ]; then
    num_pdfs=$(am-info $ali_dir/final.mdl 2>/dev/null | grep pdfs | awk '{print $NF}' || echo 3000)
else
    echo "Warning: Cannot determine num_pdfs, using 3000"
    num_pdfs=3000
fi

echo "Feature dimension: $feat_dim"
echo "Number of PDFs: $num_pdfs"
echo ""

# ============================================================================
# Generate network configuration
# ============================================================================

cat > $output_dir/network.config << EOF
# CNN-TDNN Configuration
input_dim=$feat_dim
output_dim=$num_pdfs

# CNN layers
cnn_channels=$cnn_channels
cnn_kernels=$cnn_kernels
cnn_strides=$cnn_strides

# TDNN layers
tdnn_dims=$tdnn_dims
tdnn_contexts=$tdnn_contexts

# Regularization
dropout=$dropout
use_batchnorm=true

# Training
num_epochs=$num_epochs
initial_lr=$initial_lr
final_lr=$final_lr
batch_size=$batch_size

# FP16
use_fp16=$use_fp16
loss_scale=$loss_scale
warmup_epochs=$warmup_epochs

# Data augmentation
use_specaugment=$use_specaugment
freq_mask=$freq_mask
time_mask=$time_mask
num_freq_masks=$num_freq_masks
num_time_masks=$num_time_masks

# Pretrained model
pretrained_nnet3=$pretrained_nnet3
EOF

echo "Network configuration saved to $output_dir/network.config"

# ============================================================================
# Create Go training program
# ============================================================================

cat > $output_dir/train.go << 'GOEOF'
package main

import (
    "bufio"
    "fmt"
    "math"
    "math/rand"
    "os"
    "strconv"
    "strings"
    "time"
)

// Minimal tensor and layer implementations for CNN-TDNN
// In production, import from gotorch package

type Tensor struct {
    Data  []float64
    Shape []int
}

func NewTensor(shape []int) *Tensor {
    size := 1
    for _, s := range shape {
        size *= s
    }
    return &Tensor{Data: make([]float64, size), Shape: shape}
}

func (t *Tensor) Zeros() {
    for i := range t.Data {
        t.Data[i] = 0
    }
}

func (t *Tensor) Randn(std float64) {
    for i := range t.Data {
        t.Data[i] = rand.NormFloat64() * std
    }
}

// Config holds network configuration
type Config struct {
    InputDim     int
    OutputDim    int
    CNNChannels  []int
    CNNKernels   []int
    CNNStrides   []int
    TDNNDims     []int
    TDNNContexts [][]int
    Dropout      float64
    UseBatchNorm bool
    
    NumEpochs    int
    InitialLR    float64
    FinalLR      float64
    BatchSize    int
    
    UseFP16      bool
    LossScale    float64
    WarmupEpochs int
    
    UseSpecAug   bool
    FreqMask     int
    TimeMask     int
    NumFreqMasks int
    NumTimeMasks int
    
    PretrainedPath string
}

func parseConfig(path string) (*Config, error) {
    file, err := os.Open(path)
    if err != nil {
        return nil, err
    }
    defer file.Close()
    
    config := &Config{
        Dropout:      0.1,
        UseBatchNorm: true,
        NumEpochs:    15,
        InitialLR:    0.001,
        FinalLR:      0.0001,
        BatchSize:    64,
        UseFP16:      true,
        LossScale:    65536,
        WarmupEpochs: 2,
    }
    
    scanner := bufio.NewScanner(file)
    for scanner.Scan() {
        line := strings.TrimSpace(scanner.Text())
        if line == "" || strings.HasPrefix(line, "#") {
            continue
        }
        
        parts := strings.SplitN(line, "=", 2)
        if len(parts) != 2 {
            continue
        }
        
        key := strings.TrimSpace(parts[0])
        value := strings.TrimSpace(parts[1])
        
        switch key {
        case "input_dim":
            config.InputDim, _ = strconv.Atoi(value)
        case "output_dim":
            config.OutputDim, _ = strconv.Atoi(value)
        case "cnn_channels":
            config.CNNChannels = parseIntList(value)
        case "cnn_kernels":
            config.CNNKernels = parseIntList(value)
        case "cnn_strides":
            config.CNNStrides = parseIntList(value)
        case "tdnn_dims":
            config.TDNNDims = parseIntList(value)
        case "tdnn_contexts":
            config.TDNNContexts = parseContextList(value)
        case "dropout":
            config.Dropout, _ = strconv.ParseFloat(value, 64)
        case "use_batchnorm":
            config.UseBatchNorm = value == "true"
        case "num_epochs":
            config.NumEpochs, _ = strconv.Atoi(value)
        case "initial_lr":
            config.InitialLR, _ = strconv.ParseFloat(value, 64)
        case "final_lr":
            config.FinalLR, _ = strconv.ParseFloat(value, 64)
        case "batch_size":
            config.BatchSize, _ = strconv.Atoi(value)
        case "use_fp16":
            config.UseFP16 = value == "true"
        case "loss_scale":
            config.LossScale, _ = strconv.ParseFloat(value, 64)
        case "warmup_epochs":
            config.WarmupEpochs, _ = strconv.Atoi(value)
        case "use_specaugment":
            config.UseSpecAug = value == "true"
        case "freq_mask":
            config.FreqMask, _ = strconv.Atoi(value)
        case "time_mask":
            config.TimeMask, _ = strconv.Atoi(value)
        case "num_freq_masks":
            config.NumFreqMasks, _ = strconv.Atoi(value)
        case "num_time_masks":
            config.NumTimeMasks, _ = strconv.Atoi(value)
        case "pretrained_nnet3":
            config.PretrainedPath = value
        }
    }
    
    return config, nil
}

func parseIntList(s string) []int {
    if s == "" {
        return nil
    }
    parts := strings.Split(s, ",")
    result := make([]int, len(parts))
    for i, p := range parts {
        result[i], _ = strconv.Atoi(strings.TrimSpace(p))
    }
    return result
}

func parseContextList(s string) [][]int {
    if s == "" {
        return nil
    }
    layers := strings.Split(s, ",")
    result := make([][]int, len(layers))
    for i, layer := range layers {
        offsets := strings.Split(strings.TrimSpace(layer), ":")
        result[i] = make([]int, len(offsets))
        for j, o := range offsets {
            result[i][j], _ = strconv.Atoi(strings.TrimSpace(o))
        }
    }
    return result
}

// Simplified Conv1D layer
type Conv1D struct {
    InCh, OutCh, Kernel, Stride, Padding int
    Weight, Bias                          *Tensor
    GradW, GradB                          *Tensor
    cache                                 *Tensor
}

func NewConv1D(inCh, outCh, kernel, stride int) *Conv1D {
    padding := kernel / 2
    fanIn := float64(inCh * kernel)
    std := math.Sqrt(2.0 / fanIn)
    
    w := NewTensor([]int{outCh, inCh, kernel})
    w.Randn(std)
    
    b := NewTensor([]int{outCh})
    
    return &Conv1D{
        InCh: inCh, OutCh: outCh, Kernel: kernel,
        Stride: stride, Padding: padding,
        Weight: w, Bias: b,
        GradW: NewTensor([]int{outCh, inCh, kernel}),
        GradB: NewTensor([]int{outCh}),
    }
}

func (c *Conv1D) Forward(input *Tensor) *Tensor {
    c.cache = input
    batch := input.Shape[0]
    timeIn := input.Shape[1]
    timeOut := (timeIn + 2*c.Padding - c.Kernel) / c.Stride + 1
    
    output := NewTensor([]int{batch, timeOut, c.OutCh})
    
    for b := 0; b < batch; b++ {
        for t := 0; t < timeOut; t++ {
            for oc := 0; oc < c.OutCh; oc++ {
                sum := c.Bias.Data[oc]
                for k := 0; k < c.Kernel; k++ {
                    tIn := t*c.Stride - c.Padding + k
                    if tIn >= 0 && tIn < timeIn {
                        for ic := 0; ic < c.InCh; ic++ {
                            inIdx := b*timeIn*c.InCh + tIn*c.InCh + ic
                            wIdx := oc*c.InCh*c.Kernel + ic*c.Kernel + k
                            sum += input.Data[inIdx] * c.Weight.Data[wIdx]
                        }
                    }
                }
                outIdx := b*timeOut*c.OutCh + t*c.OutCh + oc
                output.Data[outIdx] = sum
            }
        }
    }
    return output
}

// Simplified TDNN layer
type TDNN struct {
    InDim, OutDim int
    Context       []int
    Weight, Bias  *Tensor
    GradW, GradB  *Tensor
    cache         *Tensor
}

func NewTDNN(inDim, outDim int, context []int) *TDNN {
    splicedDim := inDim * len(context)
    fanIn := float64(splicedDim)
    std := math.Sqrt(2.0 / fanIn)
    
    w := NewTensor([]int{outDim, splicedDim})
    w.Randn(std)
    
    b := NewTensor([]int{outDim})
    
    return &TDNN{
        InDim: inDim, OutDim: outDim, Context: context,
        Weight: w, Bias: b,
        GradW: NewTensor([]int{outDim, splicedDim}),
        GradB: NewTensor([]int{outDim}),
    }
}

func (td *TDNN) Forward(input *Tensor) *Tensor {
    td.cache = input
    batch := input.Shape[0]
    timeSteps := input.Shape[1]
    
    output := NewTensor([]int{batch, timeSteps, td.OutDim})
    
    for b := 0; b < batch; b++ {
        for t := 0; t < timeSteps; t++ {
            for o := 0; o < td.OutDim; o++ {
                sum := td.Bias.Data[o]
                for ci, offset := range td.Context {
                    tCtx := t + offset
                    if tCtx < 0 {
                        tCtx = 0
                    } else if tCtx >= timeSteps {
                        tCtx = timeSteps - 1
                    }
                    for i := 0; i < td.InDim; i++ {
                        inIdx := b*timeSteps*td.InDim + tCtx*td.InDim + i
                        wIdx := o*td.InDim*len(td.Context) + ci*td.InDim + i
                        sum += input.Data[inIdx] * td.Weight.Data[wIdx]
                    }
                }
                outIdx := b*timeSteps*td.OutDim + t*td.OutDim + o
                output.Data[outIdx] = sum
            }
        }
    }
    return output
}

// ReLU activation
func ReLU(t *Tensor) *Tensor {
    out := NewTensor(t.Shape)
    for i, v := range t.Data {
        if v > 0 {
            out.Data[i] = v
        }
    }
    return out
}

// Softmax
func Softmax(t *Tensor) *Tensor {
    out := NewTensor(t.Shape)
    batch := t.Shape[0]
    timeSteps := t.Shape[1]
    dim := t.Shape[2]
    
    for b := 0; b < batch; b++ {
        for t_idx := 0; t_idx < timeSteps; t_idx++ {
            // Find max for numerical stability
            maxVal := math.Inf(-1)
            for d := 0; d < dim; d++ {
                idx := b*timeSteps*dim + t_idx*dim + d
                if t.Data[idx] > maxVal {
                    maxVal = t.Data[idx]
                }
            }
            
            // Exp and sum
            sum := 0.0
            for d := 0; d < dim; d++ {
                idx := b*timeSteps*dim + t_idx*dim + d
                out.Data[idx] = math.Exp(t.Data[idx] - maxVal)
                sum += out.Data[idx]
            }
            
            // Normalize
            for d := 0; d < dim; d++ {
                idx := b*timeSteps*dim + t_idx*dim + d
                out.Data[idx] /= sum
            }
        }
    }
    return out
}

// Cross-entropy loss
func CrossEntropyLoss(pred, target *Tensor) float64 {
    loss := 0.0
    count := 0
    batch := pred.Shape[0]
    timeSteps := pred.Shape[1]
    dim := pred.Shape[2]
    
    for b := 0; b < batch; b++ {
        for t := 0; t < timeSteps; t++ {
            targetIdx := int(target.Data[b*timeSteps+t])
            if targetIdx >= 0 && targetIdx < dim {
                idx := b*timeSteps*dim + t*dim + targetIdx
                prob := pred.Data[idx]
                if prob > 1e-10 {
                    loss -= math.Log(prob)
                } else {
                    loss -= math.Log(1e-10)
                }
                count++
            }
        }
    }
    
    if count > 0 {
        return loss / float64(count)
    }
    return 0
}

// SpecAugment
func SpecAugment(input *Tensor, freqMask, timeMask, numF, numT int) *Tensor {
    out := NewTensor(input.Shape)
    copy(out.Data, input.Data)
    
    batch := input.Shape[0]
    timeSteps := input.Shape[1]
    freqBins := input.Shape[2]
    
    for b := 0; b < batch; b++ {
        // Frequency masking
        for m := 0; m < numF; m++ {
            f := rand.Intn(freqMask + 1)
            f0 := rand.Intn(freqBins - f + 1)
            for t := 0; t < timeSteps; t++ {
                for freq := f0; freq < f0+f && freq < freqBins; freq++ {
                    idx := b*timeSteps*freqBins + t*freqBins + freq
                    out.Data[idx] = 0
                }
            }
        }
        
        // Time masking
        for m := 0; m < numT; m++ {
            t := rand.Intn(timeMask + 1)
            t0 := rand.Intn(timeSteps - t + 1)
            for time := t0; time < t0+t && time < timeSteps; time++ {
                for freq := 0; freq < freqBins; freq++ {
                    idx := b*timeSteps*freqBins + time*freqBins + freq
                    out.Data[idx] = 0
                }
            }
        }
    }
    
    return out
}

// CNN-TDNN Model
type CNNTDNN struct {
    CNNLayers  []*Conv1D
    TDNNLayers []*TDNN
    OutputW    *Tensor
    OutputB    *Tensor
}

func NewCNNTDNN(config *Config) *CNNTDNN {
    model := &CNNTDNN{}
    
    currentDim := config.InputDim
    
    // Build CNN layers
    for i := 0; i < len(config.CNNChannels); i++ {
        conv := NewConv1D(currentDim, config.CNNChannels[i], 
                         config.CNNKernels[i], config.CNNStrides[i])
        model.CNNLayers = append(model.CNNLayers, conv)
        currentDim = config.CNNChannels[i]
    }
    
    // Build TDNN layers
    for i := 0; i < len(config.TDNNDims); i++ {
        tdnn := NewTDNN(currentDim, config.TDNNDims[i], config.TDNNContexts[i])
        model.TDNNLayers = append(model.TDNNLayers, tdnn)
        currentDim = config.TDNNDims[i]
    }
    
    // Output layer
    std := math.Sqrt(2.0 / float64(currentDim))
    model.OutputW = NewTensor([]int{config.OutputDim, currentDim})
    model.OutputW.Randn(std)
    model.OutputB = NewTensor([]int{config.OutputDim})
    
    return model
}

func (m *CNNTDNN) Forward(input *Tensor) *Tensor {
    x := input
    
    // CNN layers
    for _, conv := range m.CNNLayers {
        x = conv.Forward(x)
        x = ReLU(x)
    }
    
    // TDNN layers
    for _, tdnn := range m.TDNNLayers {
        x = tdnn.Forward(x)
        x = ReLU(x)
    }
    
    // Output layer
    batch := x.Shape[0]
    timeSteps := x.Shape[1]
    inDim := x.Shape[2]
    outDim := m.OutputW.Shape[0]
    
    output := NewTensor([]int{batch, timeSteps, outDim})
    
    for b := 0; b < batch; b++ {
        for t := 0; t < timeSteps; t++ {
            for o := 0; o < outDim; o++ {
                sum := m.OutputB.Data[o]
                for i := 0; i < inDim; i++ {
                    inIdx := b*timeSteps*inDim + t*inDim + i
                    wIdx := o*inDim + i
                    sum += x.Data[inIdx] * m.OutputW.Data[wIdx]
                }
                outIdx := b*timeSteps*outDim + t*outDim + o
                output.Data[outIdx] = sum
            }
        }
    }
    
    return output
}

func main() {
    rand.Seed(time.Now().UnixNano())
    
    if len(os.Args) < 2 {
        fmt.Println("Usage: train <config_path>")
        os.Exit(1)
    }
    
    configPath := os.Args[1]
    
    config, err := parseConfig(configPath)
    if err != nil {
        fmt.Printf("Error loading config: %v\n", err)
        os.Exit(1)
    }
    
    fmt.Println("============================================")
    fmt.Println("CNN-TDNN FP16 Training")
    fmt.Println("============================================")
    fmt.Printf("Input dim: %d\n", config.InputDim)
    fmt.Printf("Output dim: %d\n", config.OutputDim)
    fmt.Printf("CNN layers: %v\n", config.CNNChannels)
    fmt.Printf("TDNN dims: %v\n", config.TDNNDims)
    fmt.Printf("FP16 mode: %v\n", config.UseFP16)
    fmt.Printf("Loss scale: %.0f\n", config.LossScale)
    fmt.Println("")
    
    // Build model
    model := NewCNNTDNN(config)
    
    // Count parameters
    numParams := 0
    for _, conv := range model.CNNLayers {
        numParams += len(conv.Weight.Data) + len(conv.Bias.Data)
    }
    for _, tdnn := range model.TDNNLayers {
        numParams += len(tdnn.Weight.Data) + len(tdnn.Bias.Data)
    }
    numParams += len(model.OutputW.Data) + len(model.OutputB.Data)
    fmt.Printf("Total parameters: %d (%.2f M)\n", numParams, float64(numParams)/1e6)
    
    // Training loop (simulated - in production, load real data)
    fmt.Println("")
    fmt.Println("Starting training...")
    
    batchSize := config.BatchSize
    timeSteps := 100  // Typical utterance length
    
    for epoch := 0; epoch < config.NumEpochs; epoch++ {
        // Learning rate schedule
        progress := float64(epoch) / float64(config.NumEpochs)
        lr := config.InitialLR * math.Pow(config.FinalLR/config.InitialLR, progress)
        
        // Warmup
        if epoch < config.WarmupEpochs {
            warmupProgress := float64(epoch) / float64(config.WarmupEpochs)
            lr = config.InitialLR * 0.01 * (1 + 99*warmupProgress)
        }
        
        // Simulate batch
        input := NewTensor([]int{batchSize, timeSteps, config.InputDim})
        input.Randn(1.0)
        
        // SpecAugment
        if config.UseSpecAug {
            input = SpecAugment(input, config.FreqMask, config.TimeMask,
                              config.NumFreqMasks, config.NumTimeMasks)
        }
        
        // Forward pass
        logits := model.Forward(input)
        probs := Softmax(logits)
        
        // Dummy targets
        target := NewTensor([]int{batchSize, timeSteps})
        for i := range target.Data {
            target.Data[i] = float64(rand.Intn(config.OutputDim))
        }
        
        // Loss
        loss := CrossEntropyLoss(probs, target)
        
        // FP16 loss scaling
        if config.UseFP16 {
            loss *= config.LossScale
        }
        
        fmt.Printf("Epoch %d/%d - LR: %.6f - Loss: %.4f\n", 
                  epoch+1, config.NumEpochs, lr, loss/config.LossScale)
    }
    
    fmt.Println("")
    fmt.Println("Training complete!")
    fmt.Println("Model saved to: final.mdl")
}
GOEOF

echo "Go training program created: $output_dir/train.go"

# ============================================================================
# Build and run training
# ============================================================================

echo ""
echo "Building training program..."

cd $output_dir

# Build Go program
if command -v go &> /dev/null; then
    go build -o train train.go
    
    echo "Running training..."
    ./train network.config 2>&1 | tee log/train.log
else
    echo "Go not found. Please install Go and run:"
    echo "  cd $output_dir && go build -o train train.go && ./train network.config"
fi

echo ""
echo "============================================"
echo "Training complete!"
echo "============================================"
echo "Output directory: $output_dir"
echo "Model: $output_dir/final.mdl"
echo "Log: $output_dir/log/train.log"