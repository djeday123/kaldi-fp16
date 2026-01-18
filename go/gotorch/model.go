package gotorch

import (
	"encoding/gob"
	"fmt"
	"math"
	"os"
)

// ============================================================================
// Sequential Model
// ============================================================================

type Sequential struct {
	Layers []Layer
}

func NewSequential() *Sequential {
	return &Sequential{}
}

func (s *Sequential) Add(layer Layer) *Sequential {
	s.Layers = append(s.Layers, layer)
	return s
}

func (s *Sequential) Forward(input *Tensor) *Tensor {
	x := input
	for _, layer := range s.Layers {
		x = layer.Forward(x)
	}
	return x
}

func (s *Sequential) Backward(gradOutput *Tensor) *Tensor {
	grad := gradOutput
	for i := len(s.Layers) - 1; i >= 0; i-- {
		grad = s.Layers[i].Backward(grad)
	}
	return grad
}

func (s *Sequential) Parameters() []*Tensor {
	var params []*Tensor
	for _, layer := range s.Layers {
		params = append(params, layer.Parameters()...)
	}
	return params
}

func (s *Sequential) Gradients() []*Tensor {
	var grads []*Tensor
	for _, layer := range s.Layers {
		grads = append(grads, layer.Gradients()...)
	}
	return grads
}

func (s *Sequential) ZeroGrad() {
	for _, layer := range s.Layers {
		layer.ZeroGrad()
	}
}

func (s *Sequential) NumParameters() int {
	count := 0
	for _, p := range s.Parameters() {
		if p != nil {
			count += p.Size()
		}
	}
	return count
}

func (s *Sequential) Summary() {
	fmt.Println("Model Summary")
	fmt.Println("============================================================")
	totalParams := 0
	for i, layer := range s.Layers {
		params := 0
		for _, p := range layer.Parameters() {
			if p != nil {
				params += p.Size()
			}
		}
		totalParams += params
		fmt.Printf("%2d. %-20s  params: %d\n", i+1, layer.Name(), params)
	}
	fmt.Println("============================================================")
	fmt.Printf("Total parameters: %d (%.2f M)\n", totalParams, float64(totalParams)/1e6)
}

func (s *Sequential) SetTraining(training bool) {
	for _, layer := range s.Layers {
		if bn, ok := layer.(*BatchNormLayer); ok {
			bn.Training = training
		}
		if do, ok := layer.(*DropoutLayer); ok {
			do.Training = training
		}
	}
}

// ============================================================================
// Kaldi DNN Builder
// ============================================================================

type KaldiDNNConfig struct {
	InputDim     int
	OutputDim    int
	HiddenDims   []int
	Activation   string
	UseDropout   bool
	DropoutRate  float64
	UseBatchNorm bool
}

func BuildKaldiDNN(config *KaldiDNNConfig) *Sequential {
	model := NewSequential()

	prevDim := config.InputDim

	for i, hiddenDim := range config.HiddenDims {
		affine := NewAffineLayer(prevDim, hiddenDim)
		affine.name = fmt.Sprintf("affine%d", i+1)
		model.Add(affine)

		if config.UseBatchNorm {
			bn := NewBatchNormLayer(hiddenDim)
			bn.name = fmt.Sprintf("bn%d", i+1)
			model.Add(bn)
		}

		relu := &ReLULayer{name: fmt.Sprintf("relu%d", i+1)}
		model.Add(relu)

		if config.UseDropout && config.DropoutRate > 0 {
			dropout := NewDropoutLayer(config.DropoutRate)
			dropout.name = fmt.Sprintf("dropout%d", i+1)
			model.Add(dropout)
		}

		prevDim = hiddenDim
	}

	output := NewAffineLayer(prevDim, config.OutputDim)
	output.name = "output"
	model.Add(output)

	return model
}

func BuildKaldiTDNN(inputDim, outputDim int) *Sequential {
	model := NewSequential()

	// Layer 1: context [-2, -1, 0, 1, 2]
	tdnn1 := NewTDNNLayer(inputDim, 512, []int{-2, -1, 0, 1, 2})
	tdnn1.name = "tdnn1"
	model.Add(tdnn1)
	model.Add(&ReLULayer{name: "relu1"})
	bn1 := NewBatchNormLayer(512)
	bn1.name = "bn1"
	model.Add(bn1)

	// Layer 2: context [-2, 0, 2]
	tdnn2 := NewTDNNLayer(512, 512, []int{-2, 0, 2})
	tdnn2.name = "tdnn2"
	model.Add(tdnn2)
	model.Add(&ReLULayer{name: "relu2"})
	bn2 := NewBatchNormLayer(512)
	bn2.name = "bn2"
	model.Add(bn2)

	// Layer 3: context [-3, 0, 3]
	tdnn3 := NewTDNNLayer(512, 512, []int{-3, 0, 3})
	tdnn3.name = "tdnn3"
	model.Add(tdnn3)
	model.Add(&ReLULayer{name: "relu3"})
	bn3 := NewBatchNormLayer(512)
	bn3.name = "bn3"
	model.Add(bn3)

	// Dense
	dense1 := NewAffineLayer(512, 512)
	dense1.name = "dense1"
	model.Add(dense1)
	model.Add(&ReLULayer{name: "relu4"})

	dense2 := NewAffineLayer(512, 512)
	dense2.name = "dense2"
	model.Add(dense2)
	model.Add(&ReLULayer{name: "relu5"})

	// Output
	output := NewAffineLayer(512, outputDim)
	output.name = "output"
	model.Add(output)

	return model
}

// ============================================================================
// Optimizers
// ============================================================================

type Optimizer interface {
	Step()
	ZeroGrad()
}

// SGD optimizer
type SGD struct {
	Params      []*Tensor
	Grads       []*Tensor
	LR          float64
	Momentum    float64
	WeightDecay float64
	velocities  []*Tensor
}

func NewSGD(params, grads []*Tensor, lr float64) *SGD {
	velocities := make([]*Tensor, len(params))
	for i, p := range params {
		if p != nil {
			velocities[i] = Zeros(p.Shape)
		}
	}

	return &SGD{
		Params:      params,
		Grads:       grads,
		LR:          lr,
		Momentum:    0.9,
		WeightDecay: 0.0,
		velocities:  velocities,
	}
}

func (o *SGD) Step() {
	for i, p := range o.Params {
		if p == nil || o.Grads[i] == nil {
			continue
		}

		g := o.Grads[i]

		// Weight decay
		if o.WeightDecay > 0 {
			for j := range g.Data {
				g.Data[j] += o.WeightDecay * p.Data[j]
			}
		}

		// Momentum
		if o.Momentum > 0 {
			v := o.velocities[i]
			for j := range v.Data {
				v.Data[j] = o.Momentum*v.Data[j] + g.Data[j]
			}
			for j := range p.Data {
				p.Data[j] -= o.LR * v.Data[j]
			}
		} else {
			for j := range p.Data {
				p.Data[j] -= o.LR * g.Data[j]
			}
		}
	}
}

func (o *SGD) ZeroGrad() {
	for _, g := range o.Grads {
		if g != nil {
			for i := range g.Data {
				g.Data[i] = 0
			}
		}
	}
}

// Adam optimizer
type Adam struct {
	Params      []*Tensor
	Grads       []*Tensor
	LR          float64
	Beta1       float64
	Beta2       float64
	Eps         float64
	WeightDecay float64
	m           []*Tensor
	v           []*Tensor
	t           int
}

func NewAdam(params, grads []*Tensor, lr float64) *Adam {
	m := make([]*Tensor, len(params))
	v := make([]*Tensor, len(params))

	for i, p := range params {
		if p != nil {
			m[i] = Zeros(p.Shape)
			v[i] = Zeros(p.Shape)
		}
	}

	return &Adam{
		Params:      params,
		Grads:       grads,
		LR:          lr,
		Beta1:       0.9,
		Beta2:       0.999,
		Eps:         1e-8,
		WeightDecay: 0.0,
		m:           m,
		v:           v,
		t:           0,
	}
}

func (o *Adam) Step() {
	o.t++

	for i, p := range o.Params {
		if p == nil || o.Grads[i] == nil {
			continue
		}

		g := o.Grads[i]
		m := o.m[i]
		v := o.v[i]

		for j := range g.Data {
			grad := g.Data[j]
			if o.WeightDecay > 0 {
				grad += o.WeightDecay * p.Data[j]
			}

			m.Data[j] = o.Beta1*m.Data[j] + (1-o.Beta1)*grad
			v.Data[j] = o.Beta2*v.Data[j] + (1-o.Beta2)*grad*grad

			mHat := m.Data[j] / (1 - math.Pow(o.Beta1, float64(o.t)))
			vHat := v.Data[j] / (1 - math.Pow(o.Beta2, float64(o.t)))

			p.Data[j] -= o.LR * mHat / (math.Sqrt(vHat) + o.Eps)
		}
	}
}

func (o *Adam) ZeroGrad() {
	for _, g := range o.Grads {
		if g != nil {
			for i := range g.Data {
				g.Data[i] = 0
			}
		}
	}
}

// ============================================================================
// Learning Rate Schedulers
// ============================================================================

type LRScheduler interface {
	Step()
	GetLR() float64
}

type StepLR struct {
	Optimizer Optimizer
	StepSize  int
	Gamma     float64
	baseLR    float64
	epoch     int
}

func NewStepLR(opt *SGD, stepSize int, gamma float64) *StepLR {
	return &StepLR{
		Optimizer: opt,
		StepSize:  stepSize,
		Gamma:     gamma,
		baseLR:    opt.LR,
		epoch:     0,
	}
}

func (s *StepLR) Step() {
	s.epoch++
	if s.epoch%s.StepSize == 0 {
		if sgd, ok := s.Optimizer.(*SGD); ok {
			sgd.LR *= s.Gamma
		}
	}
}

func (s *StepLR) GetLR() float64 {
	if sgd, ok := s.Optimizer.(*SGD); ok {
		return sgd.LR
	}
	return s.baseLR
}

type ExponentialLR struct {
	Optimizer *SGD
	Gamma     float64
	baseLR    float64
}

func NewExponentialLR(opt *SGD, gamma float64) *ExponentialLR {
	return &ExponentialLR{
		Optimizer: opt,
		Gamma:     gamma,
		baseLR:    opt.LR,
	}
}

func (e *ExponentialLR) Step() {
	e.Optimizer.LR *= e.Gamma
}

func (e *ExponentialLR) GetLR() float64 {
	return e.Optimizer.LR
}

// ============================================================================
// Model Save/Load
// ============================================================================

type ModelData struct {
	Params [][]float64
	Shapes [][]int
}

func SaveModel(model *Sequential, path string) error {
	params := model.Parameters()

	data := ModelData{
		Params: make([][]float64, len(params)),
		Shapes: make([][]int, len(params)),
	}

	for i, p := range params {
		if p != nil {
			data.Params[i] = make([]float64, len(p.Data))
			copy(data.Params[i], p.Data)
			data.Shapes[i] = append([]int{}, p.Shape...)
		}
	}

	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	return encoder.Encode(data)
}

func LoadModel(model *Sequential, path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	var data ModelData
	decoder := gob.NewDecoder(file)
	if err := decoder.Decode(&data); err != nil {
		return err
	}

	params := model.Parameters()
	for i, p := range params {
		if p != nil && i < len(data.Params) && data.Params[i] != nil {
			copy(p.Data, data.Params[i])
		}
	}

	return nil
}
