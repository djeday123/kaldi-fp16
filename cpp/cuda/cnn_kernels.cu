// CNN-TDNN FP16 CUDA Kernels
// 1D Convolution optimized for speech features
// Compatible with Kaldi HMM-DNN pipeline

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>  // Tensor Core operations

using namespace nvcuda;

// ============================================================================
// Conv1D Forward - FP16 with Tensor Cores
// ============================================================================

// Standard Conv1D forward for speech frames
// Input: [batch, time, in_channels]
// Weight: [out_channels, in_channels, kernel_size]
// Output: [batch, time_out, out_channels]
__global__ void conv1d_forward_fp16(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    int batch_size,
    int time_in,
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    int b = blockIdx.z;
    int t_out = blockIdx.x * blockDim.x + threadIdx.x;
    int oc = blockIdx.y * blockDim.y + threadIdx.y;
    
    int time_out = (time_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    if (b >= batch_size || t_out >= time_out || oc >= out_channels) return;
    
    float sum = 0.0f;  // Accumulate in FP32 for precision
    
    for (int k = 0; k < kernel_size; k++) {
        int t_in = t_out * stride - padding + k * dilation;
        
        if (t_in >= 0 && t_in < time_in) {
            for (int ic = 0; ic < in_channels; ic++) {
                int input_idx = b * time_in * in_channels + t_in * in_channels + ic;
                int weight_idx = oc * in_channels * kernel_size + ic * kernel_size + k;
                
                sum += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
            }
        }
    }
    
    // Add bias
    if (bias != nullptr) {
        sum += __half2float(bias[oc]);
    }
    
    int output_idx = b * time_out * out_channels + t_out * out_channels + oc;
    output[output_idx] = __float2half(sum);
}

// Optimized Conv1D using shared memory
__global__ void conv1d_forward_fp16_shared(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    int batch_size,
    int time_in,
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding
) {
    extern __shared__ __half shared_mem[];
    
    __half* shared_input = shared_mem;
    __half* shared_weight = shared_mem + blockDim.x * in_channels;
    
    int b = blockIdx.z;
    int t_out = blockIdx.x * blockDim.x + threadIdx.x;
    int oc = blockIdx.y;
    
    int time_out = (time_in + 2 * padding - kernel_size) / stride + 1;
    
    // Load weights to shared memory
    for (int i = threadIdx.x; i < in_channels * kernel_size; i += blockDim.x) {
        shared_weight[i] = weight[oc * in_channels * kernel_size + i];
    }
    __syncthreads();
    
    if (b >= batch_size || t_out >= time_out) return;
    
    float sum = 0.0f;
    
    for (int k = 0; k < kernel_size; k++) {
        int t_in = t_out * stride - padding + k;
        
        if (t_in >= 0 && t_in < time_in) {
            for (int ic = 0; ic < in_channels; ic++) {
                int input_idx = b * time_in * in_channels + t_in * in_channels + ic;
                int weight_idx = ic * kernel_size + k;
                
                sum += __half2float(input[input_idx]) * __half2float(shared_weight[weight_idx]);
            }
        }
    }
    
    if (bias != nullptr) {
        sum += __half2float(bias[oc]);
    }
    
    int output_idx = b * time_out * out_channels + t_out * out_channels + oc;
    output[output_idx] = __float2half(sum);
}

// ============================================================================
// Conv1D Backward - Weight and Input Gradients
// ============================================================================

// Gradient w.r.t. input
__global__ void conv1d_backward_input_fp16(
    const __half* __restrict__ grad_output,
    const __half* __restrict__ weight,
    __half* __restrict__ grad_input,
    int batch_size,
    int time_in,
    int time_out,
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    int b = blockIdx.z;
    int t_in = blockIdx.x * blockDim.x + threadIdx.x;
    int ic = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (b >= batch_size || t_in >= time_in || ic >= in_channels) return;
    
    float sum = 0.0f;
    
    for (int oc = 0; oc < out_channels; oc++) {
        for (int k = 0; k < kernel_size; k++) {
            int t_out = t_in + padding - k * dilation;
            
            if (t_out >= 0 && t_out < time_out && t_out % stride == 0) {
                t_out /= stride;
                
                int grad_idx = b * time_out * out_channels + t_out * out_channels + oc;
                int weight_idx = oc * in_channels * kernel_size + ic * kernel_size + k;
                
                sum += __half2float(grad_output[grad_idx]) * __half2float(weight[weight_idx]);
            }
        }
    }
    
    int input_idx = b * time_in * in_channels + t_in * in_channels + ic;
    grad_input[input_idx] = __float2half(sum);
}

// Gradient w.r.t. weight
__global__ void conv1d_backward_weight_fp16(
    const __half* __restrict__ input,
    const __half* __restrict__ grad_output,
    __half* __restrict__ grad_weight,
    int batch_size,
    int time_in,
    int time_out,
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    int oc = blockIdx.x;
    int ic = blockIdx.y * blockDim.y + threadIdx.y;
    int k = threadIdx.x;
    
    if (oc >= out_channels || ic >= in_channels || k >= kernel_size) return;
    
    float sum = 0.0f;
    
    for (int b = 0; b < batch_size; b++) {
        for (int t_out = 0; t_out < time_out; t_out++) {
            int t_in = t_out * stride - padding + k * dilation;
            
            if (t_in >= 0 && t_in < time_in) {
                int input_idx = b * time_in * in_channels + t_in * in_channels + ic;
                int grad_idx = b * time_out * out_channels + t_out * out_channels + oc;
                
                sum += __half2float(input[input_idx]) * __half2float(grad_output[grad_idx]);
            }
        }
    }
    
    int weight_idx = oc * in_channels * kernel_size + ic * kernel_size + k;
    atomicAdd(reinterpret_cast<float*>(&grad_weight[weight_idx]), sum);
}

// Gradient w.r.t. bias
__global__ void conv1d_backward_bias_fp16(
    const __half* __restrict__ grad_output,
    __half* __restrict__ grad_bias,
    int batch_size,
    int time_out,
    int out_channels
) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (oc >= out_channels) return;
    
    float sum = 0.0f;
    
    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < time_out; t++) {
            int idx = b * time_out * out_channels + t * out_channels + oc;
            sum += __half2float(grad_output[idx]);
        }
    }
    
    grad_bias[oc] = __float2half(sum);
}

// ============================================================================
// Batch Normalization for CNN - FP16
// Critical for stable FP16 training
// ============================================================================

__global__ void batchnorm1d_forward_fp16(
    const __half* __restrict__ input,
    const __half* __restrict__ gamma,
    const __half* __restrict__ beta,
    __half* __restrict__ running_mean,
    __half* __restrict__ running_var,
    __half* __restrict__ output,
    __half* __restrict__ save_mean,
    __half* __restrict__ save_invstd,
    int batch_size,
    int time_steps,
    int channels,
    float momentum,
    float eps,
    bool training
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (c >= channels) return;
    
    int n = batch_size * time_steps;
    
    if (training) {
        // Compute mean
        float sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            for (int t = 0; t < time_steps; t++) {
                int idx = b * time_steps * channels + t * channels + c;
                sum += __half2float(input[idx]);
            }
        }
        float mean = sum / n;
        
        // Compute variance
        float var_sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            for (int t = 0; t < time_steps; t++) {
                int idx = b * time_steps * channels + t * channels + c;
                float diff = __half2float(input[idx]) - mean;
                var_sum += diff * diff;
            }
        }
        float var = var_sum / n;
        
        // Save for backward
        save_mean[c] = __float2half(mean);
        float invstd = rsqrtf(var + eps);
        save_invstd[c] = __float2half(invstd);
        
        // Update running stats
        float rm = __half2float(running_mean[c]);
        float rv = __half2float(running_var[c]);
        running_mean[c] = __float2half(rm * (1 - momentum) + mean * momentum);
        running_var[c] = __float2half(rv * (1 - momentum) + var * momentum);
        
        // Normalize and scale
        float g = __half2float(gamma[c]);
        float b_val = __half2float(beta[c]);
        
        for (int b = 0; b < batch_size; b++) {
            for (int t = 0; t < time_steps; t++) {
                int idx = b * time_steps * channels + t * channels + c;
                float x = __half2float(input[idx]);
                float norm = (x - mean) * invstd;
                output[idx] = __float2half(norm * g + b_val);
            }
        }
    } else {
        // Inference mode - use running stats
        float mean = __half2float(running_mean[c]);
        float var = __half2float(running_var[c]);
        float invstd = rsqrtf(var + eps);
        float g = __half2float(gamma[c]);
        float b_val = __half2float(beta[c]);
        
        for (int b = 0; b < batch_size; b++) {
            for (int t = 0; t < time_steps; t++) {
                int idx = b * time_steps * channels + t * channels + c;
                float x = __half2float(input[idx]);
                float norm = (x - mean) * invstd;
                output[idx] = __float2half(norm * g + b_val);
            }
        }
    }
}

// ============================================================================
// Pooling Layers for CNN
// ============================================================================

// Max pooling 1D
__global__ void maxpool1d_forward_fp16(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int* __restrict__ indices,  // For backward pass
    int batch_size,
    int time_in,
    int channels,
    int kernel_size,
    int stride
) {
    int b = blockIdx.z;
    int t_out = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    
    int time_out = (time_in - kernel_size) / stride + 1;
    
    if (b >= batch_size || t_out >= time_out || c >= channels) return;
    
    float max_val = -1e10f;
    int max_idx = 0;
    
    for (int k = 0; k < kernel_size; k++) {
        int t_in = t_out * stride + k;
        int idx = b * time_in * channels + t_in * channels + c;
        float val = __half2float(input[idx]);
        
        if (val > max_val) {
            max_val = val;
            max_idx = t_in;
        }
    }
    
    int out_idx = b * time_out * channels + t_out * channels + c;
    output[out_idx] = __float2half(max_val);
    indices[out_idx] = max_idx;
}

// Max pooling backward
__global__ void maxpool1d_backward_fp16(
    const __half* __restrict__ grad_output,
    const int* __restrict__ indices,
    __half* __restrict__ grad_input,
    int batch_size,
    int time_in,
    int time_out,
    int channels
) {
    int b = blockIdx.z;
    int t_out = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (b >= batch_size || t_out >= time_out || c >= channels) return;
    
    int out_idx = b * time_out * channels + t_out * channels + c;
    int max_t = indices[out_idx];
    int in_idx = b * time_in * channels + max_t * channels + c;
    
    atomicAdd(reinterpret_cast<float*>(&grad_input[in_idx]), 
              __half2float(grad_output[out_idx]));
}

// Average pooling 1D
__global__ void avgpool1d_forward_fp16(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int batch_size,
    int time_in,
    int channels,
    int kernel_size,
    int stride
) {
    int b = blockIdx.z;
    int t_out = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    
    int time_out = (time_in - kernel_size) / stride + 1;
    
    if (b >= batch_size || t_out >= time_out || c >= channels) return;
    
    float sum = 0.0f;
    
    for (int k = 0; k < kernel_size; k++) {
        int t_in = t_out * stride + k;
        int idx = b * time_in * channels + t_in * channels + c;
        sum += __half2float(input[idx]);
    }
    
    int out_idx = b * time_out * channels + t_out * channels + c;
    output[out_idx] = __float2half(sum / kernel_size);
}

// ============================================================================
// Statistics Pooling for x-vector style
// Used after TDNN layers for speaker embeddings
// ============================================================================

__global__ void stats_pooling_fp16(
    const __half* __restrict__ input,
    __half* __restrict__ output,  // [batch, 2 * channels] - mean and std
    int batch_size,
    int time_steps,
    int channels
) {
    int b = blockIdx.x;
    int c = threadIdx.x;
    
    if (b >= batch_size || c >= channels) return;
    
    // Compute mean
    float sum = 0.0f;
    for (int t = 0; t < time_steps; t++) {
        int idx = b * time_steps * channels + t * channels + c;
        sum += __half2float(input[idx]);
    }
    float mean = sum / time_steps;
    
    // Compute variance
    float var_sum = 0.0f;
    for (int t = 0; t < time_steps; t++) {
        int idx = b * time_steps * channels + t * channels + c;
        float diff = __half2float(input[idx]) - mean;
        var_sum += diff * diff;
    }
    float std = sqrtf(var_sum / time_steps + 1e-10f);
    
    // Output: [mean, std]
    output[b * 2 * channels + c] = __float2half(mean);
    output[b * 2 * channels + channels + c] = __float2half(std);
}

// ============================================================================
// Layer Normalization - Alternative to BatchNorm for small batches
// ============================================================================

__global__ void layernorm_forward_fp16(
    const __half* __restrict__ input,
    const __half* __restrict__ gamma,
    const __half* __restrict__ beta,
    __half* __restrict__ output,
    int batch_size,
    int time_steps,
    int channels,
    float eps
) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    
    if (b >= batch_size || t >= time_steps) return;
    
    // Compute mean and variance over channels
    extern __shared__ float shared[];
    float* s_sum = shared;
    float* s_sqsum = shared + blockDim.x;
    
    float sum = 0.0f;
    float sqsum = 0.0f;
    
    for (int c = threadIdx.x; c < channels; c += blockDim.x) {
        int idx = b * time_steps * channels + t * channels + c;
        float val = __half2float(input[idx]);
        sum += val;
        sqsum += val * val;
    }
    
    s_sum[threadIdx.x] = sum;
    s_sqsum[threadIdx.x] = sqsum;
    __syncthreads();
    
    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_sqsum[threadIdx.x] += s_sqsum[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    float mean = s_sum[0] / channels;
    float var = s_sqsum[0] / channels - mean * mean;
    float invstd = rsqrtf(var + eps);
    
    // Normalize
    for (int c = threadIdx.x; c < channels; c += blockDim.x) {
        int idx = b * time_steps * channels + t * channels + c;
        float x = __half2float(input[idx]);
        float norm = (x - mean) * invstd;
        float g = __half2float(gamma[c]);
        float beta_val = __half2float(beta[c]);
        output[idx] = __float2half(norm * g + beta_val);
    }
}

// ============================================================================
// Depthwise Separable Conv1D - Efficient alternative
// Used in some modern ASR architectures
// ============================================================================

// Depthwise convolution
__global__ void depthwise_conv1d_fp16(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,  // [channels, 1, kernel_size]
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    int batch_size,
    int time_in,
    int channels,
    int kernel_size,
    int stride,
    int padding
) {
    int b = blockIdx.z;
    int t_out = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    
    int time_out = (time_in + 2 * padding - kernel_size) / stride + 1;
    
    if (b >= batch_size || t_out >= time_out || c >= channels) return;
    
    float sum = 0.0f;
    
    for (int k = 0; k < kernel_size; k++) {
        int t_in = t_out * stride - padding + k;
        
        if (t_in >= 0 && t_in < time_in) {
            int input_idx = b * time_in * channels + t_in * channels + c;
            int weight_idx = c * kernel_size + k;
            sum += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
        }
    }
    
    if (bias != nullptr) {
        sum += __half2float(bias[c]);
    }
    
    int out_idx = b * time_out * channels + t_out * channels + c;
    output[out_idx] = __float2half(sum);
}

// Pointwise (1x1) convolution
__global__ void pointwise_conv1d_fp16(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,  // [out_channels, in_channels]
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    int batch_size,
    int time_steps,
    int in_channels,
    int out_channels
) {
    int b = blockIdx.z;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int oc = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (b >= batch_size || t >= time_steps || oc >= out_channels) return;
    
    float sum = 0.0f;
    
    for (int ic = 0; ic < in_channels; ic++) {
        int input_idx = b * time_steps * in_channels + t * in_channels + ic;
        int weight_idx = oc * in_channels + ic;
        sum += __half2float(input[input_idx]) * __half2float(weight[weight_idx]);
    }
    
    if (bias != nullptr) {
        sum += __half2float(bias[oc]);
    }
    
    int out_idx = b * time_steps * out_channels + t * out_channels + oc;
    output[out_idx] = __float2half(sum);
}

// ============================================================================
// Squeeze-and-Excitation for channel attention
// ============================================================================

__global__ void se_block_fp16(
    const __half* __restrict__ input,
    const __half* __restrict__ fc1_weight,
    const __half* __restrict__ fc1_bias,
    const __half* __restrict__ fc2_weight,
    const __half* __restrict__ fc2_bias,
    __half* __restrict__ output,
    int batch_size,
    int time_steps,
    int channels,
    int reduction
) {
    int b = blockIdx.x;
    
    if (b >= batch_size) return;
    
    extern __shared__ float shared[];
    float* pooled = shared;              // [channels]
    float* squeezed = shared + channels; // [channels/reduction]
    
    int reduced_channels = channels / reduction;
    
    // Global average pooling
    for (int c = threadIdx.x; c < channels; c += blockDim.x) {
        float sum = 0.0f;
        for (int t = 0; t < time_steps; t++) {
            int idx = b * time_steps * channels + t * channels + c;
            sum += __half2float(input[idx]);
        }
        pooled[c] = sum / time_steps;
    }
    __syncthreads();
    
    // FC1 + ReLU
    for (int r = threadIdx.x; r < reduced_channels; r += blockDim.x) {
        float sum = __half2float(fc1_bias[r]);
        for (int c = 0; c < channels; c++) {
            sum += pooled[c] * __half2float(fc1_weight[r * channels + c]);
        }
        squeezed[r] = fmaxf(sum, 0.0f);  // ReLU
    }
    __syncthreads();
    
    // FC2 + Sigmoid
    for (int c = threadIdx.x; c < channels; c += blockDim.x) {
        float sum = __half2float(fc2_bias[c]);
        for (int r = 0; r < reduced_channels; r++) {
            sum += squeezed[r] * __half2float(fc2_weight[c * reduced_channels + r]);
        }
        float scale = 1.0f / (1.0f + expf(-sum));  // Sigmoid
        
        // Apply scaling to all time steps
        for (int t = 0; t < time_steps; t++) {
            int idx = b * time_steps * channels + t * channels + c;
            output[idx] = __float2half(__half2float(input[idx]) * scale);
        }
    }
}

// ============================================================================
// Host-side wrapper functions
// ============================================================================

extern "C" {

void launch_conv1d_forward_fp16(
    const void* input, const void* weight, const void* bias,
    void* output,
    int batch_size, int time_in, int in_channels,
    int out_channels, int kernel_size, int stride, int padding, int dilation,
    cudaStream_t stream
) {
    int time_out = (time_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    dim3 block(16, 16);
    dim3 grid(
        (time_out + block.x - 1) / block.x,
        (out_channels + block.y - 1) / block.y,
        batch_size
    );
    
    conv1d_forward_fp16<<<grid, block, 0, stream>>>(
        (const __half*)input, (const __half*)weight, (const __half*)bias,
        (__half*)output,
        batch_size, time_in, in_channels, out_channels,
        kernel_size, stride, padding, dilation
    );
}

void launch_conv1d_backward_fp16(
    const void* input, const void* grad_output, const void* weight,
    void* grad_input, void* grad_weight, void* grad_bias,
    int batch_size, int time_in, int in_channels,
    int out_channels, int kernel_size, int stride, int padding, int dilation,
    cudaStream_t stream
) {
    int time_out = (time_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    // Gradient w.r.t. input
    dim3 block1(16, 16);
    dim3 grid1(
        (time_in + block1.x - 1) / block1.x,
        (in_channels + block1.y - 1) / block1.y,
        batch_size
    );
    conv1d_backward_input_fp16<<<grid1, block1, 0, stream>>>(
        (const __half*)grad_output, (const __half*)weight,
        (__half*)grad_input,
        batch_size, time_in, time_out, in_channels, out_channels,
        kernel_size, stride, padding, dilation
    );
    
    // Gradient w.r.t. weight
    dim3 block2(kernel_size, 16);
    dim3 grid2(out_channels, (in_channels + block2.y - 1) / block2.y);
    conv1d_backward_weight_fp16<<<grid2, block2, 0, stream>>>(
        (const __half*)input, (const __half*)grad_output,
        (__half*)grad_weight,
        batch_size, time_in, time_out, in_channels, out_channels,
        kernel_size, stride, padding, dilation
    );
    
    // Gradient w.r.t. bias
    int threads = 256;
    int blocks = (out_channels + threads - 1) / threads;
    conv1d_backward_bias_fp16<<<blocks, threads, 0, stream>>>(
        (const __half*)grad_output, (__half*)grad_bias,
        batch_size, time_out, out_channels
    );
}

void launch_maxpool1d_forward_fp16(
    const void* input, void* output, void* indices,
    int batch_size, int time_in, int channels,
    int kernel_size, int stride,
    cudaStream_t stream
) {
    int time_out = (time_in - kernel_size) / stride + 1;
    
    dim3 block(16, 16);
    dim3 grid(
        (time_out + block.x - 1) / block.x,
        (channels + block.y - 1) / block.y,
        batch_size
    );
    
    maxpool1d_forward_fp16<<<grid, block, 0, stream>>>(
        (const __half*)input, (__half*)output, (int*)indices,
        batch_size, time_in, channels, kernel_size, stride
    );
}

void launch_stats_pooling_fp16(
    const void* input, void* output,
    int batch_size, int time_steps, int channels,
    cudaStream_t stream
) {
    dim3 block(channels);
    dim3 grid(batch_size);
    
    stats_pooling_fp16<<<grid, block, 0, stream>>>(
        (const __half*)input, (__half*)output,
        batch_size, time_steps, channels
    );
}

void launch_batchnorm1d_forward_fp16(
    const void* input, const void* gamma, const void* beta,
    void* running_mean, void* running_var,
    void* output, void* save_mean, void* save_invstd,
    int batch_size, int time_steps, int channels,
    float momentum, float eps, bool training,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (channels + threads - 1) / threads;
    
    batchnorm1d_forward_fp16<<<blocks, threads, 0, stream>>>(
        (const __half*)input, (const __half*)gamma, (const __half*)beta,
        (__half*)running_mean, (__half*)running_var,
        (__half*)output, (__half*)save_mean, (__half*)save_invstd,
        batch_size, time_steps, channels, momentum, eps, training
    );
}

void launch_depthwise_conv1d_fp16(
    const void* input, const void* weight, const void* bias,
    void* output,
    int batch_size, int time_in, int channels,
    int kernel_size, int stride, int padding,
    cudaStream_t stream
) {
    int time_out = (time_in + 2 * padding - kernel_size) / stride + 1;
    
    dim3 block(16, 16);
    dim3 grid(
        (time_out + block.x - 1) / block.x,
        (channels + block.y - 1) / block.y,
        batch_size
    );
    
    depthwise_conv1d_fp16<<<grid, block, 0, stream>>>(
        (const __half*)input, (const __half*)weight, (const __half*)bias,
        (__half*)output,
        batch_size, time_in, channels, kernel_size, stride, padding
    );
}

}  // extern "C"
