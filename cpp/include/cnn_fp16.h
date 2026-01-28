// CNN-TDNN FP16 Operations Header
// Provides C interface for CNN operations with Tensor Cores

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Conv1D Operations
// ============================================================================

// Forward pass for 1D convolution
// Input:  [batch, time_in, in_channels]  - FP16
// Weight: [out_channels, in_channels, kernel_size] - FP16
// Bias:   [out_channels] - FP16 (can be NULL)
// Output: [batch, time_out, out_channels] - FP16
void launch_conv1d_forward_fp16(
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    int batch_size,
    int time_in,
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    cudaStream_t stream
);

// Backward pass for 1D convolution
void launch_conv1d_backward_fp16(
    const void* input,
    const void* grad_output,
    const void* weight,
    void* grad_input,
    void* grad_weight,
    void* grad_bias,
    int batch_size,
    int time_in,
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    cudaStream_t stream
);

// ============================================================================
// Pooling Operations
// ============================================================================

// Max pooling 1D forward
void launch_maxpool1d_forward_fp16(
    const void* input,
    void* output,
    void* indices,  // int32 tensor for backward pass
    int batch_size,
    int time_in,
    int channels,
    int kernel_size,
    int stride,
    cudaStream_t stream
);

// Max pooling 1D backward
void launch_maxpool1d_backward_fp16(
    const void* grad_output,
    const void* indices,
    void* grad_input,
    int batch_size,
    int time_in,
    int time_out,
    int channels,
    cudaStream_t stream
);

// Statistics pooling (mean + std) for x-vector
// Input:  [batch, time, channels]
// Output: [batch, 2 * channels]
void launch_stats_pooling_fp16(
    const void* input,
    void* output,
    int batch_size,
    int time_steps,
    int channels,
    cudaStream_t stream
);

// ============================================================================
// Normalization
// ============================================================================

// Batch normalization 1D
void launch_batchnorm1d_forward_fp16(
    const void* input,
    const void* gamma,
    const void* beta,
    void* running_mean,
    void* running_var,
    void* output,
    void* save_mean,
    void* save_invstd,
    int batch_size,
    int time_steps,
    int channels,
    float momentum,
    float eps,
    bool training,
    cudaStream_t stream
);

// Layer normalization
void launch_layernorm_forward_fp16(
    const void* input,
    const void* gamma,
    const void* beta,
    void* output,
    int batch_size,
    int time_steps,
    int channels,
    float eps,
    cudaStream_t stream
);

// ============================================================================
// Depthwise Separable Convolution
// ============================================================================

// Depthwise convolution (each channel independently)
void launch_depthwise_conv1d_fp16(
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    int batch_size,
    int time_in,
    int channels,
    int kernel_size,
    int stride,
    int padding,
    cudaStream_t stream
);

// Pointwise convolution (1x1 conv)
void launch_pointwise_conv1d_fp16(
    const void* input,
    const void* weight,
    const void* bias,
    void* output,
    int batch_size,
    int time_steps,
    int in_channels,
    int out_channels,
    cudaStream_t stream
);

// ============================================================================
// Utility Functions
// ============================================================================

// Calculate output time dimension for conv1d
static inline int conv1d_output_size(int time_in, int kernel_size, int stride, int padding, int dilation) {
    return (time_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
}

// Calculate output time dimension for pooling
static inline int pool1d_output_size(int time_in, int kernel_size, int stride) {
    return (time_in - kernel_size) / stride + 1;
}

#ifdef __cplusplus
}
#endif

// ============================================================================
// C++ Interface (when not using C linkage)
// ============================================================================

#ifdef __cplusplus

namespace kaldi_fp16 {
namespace cnn {

// Conv1D configuration
struct Conv1DConfig {
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride = 1;
    int padding = 0;
    int dilation = 1;
    bool use_bias = true;
};

// TDNN configuration (affine with frame splicing)
struct TDNNConfig {
    int input_dim;
    int output_dim;
    std::vector<int> context;  // e.g., {-2, -1, 0, 1, 2}
    bool use_batchnorm = true;
    float dropout = 0.0f;
};

// CNN-TDNN model configuration
struct CNNTDNNConfig {
    int input_dim;    // Feature dimension (e.g., 40 for MFCC)
    int output_dim;   // Number of PDFs
    
    // CNN layers
    std::vector<Conv1DConfig> cnn_layers;
    
    // TDNN layers
    std::vector<TDNNConfig> tdnn_layers;
    
    // Regularization
    float dropout = 0.1f;
    bool use_batchnorm = true;
    
    // Training
    bool use_fp16 = true;
    float initial_loss_scale = 65536.0f;
};

// Factory function to create standard CNN-TDNN config
inline CNNTDNNConfig make_standard_cnn_tdnn(int input_dim, int output_dim) {
    CNNTDNNConfig config;
    config.input_dim = input_dim;
    config.output_dim = output_dim;
    
    // CNN front-end
    config.cnn_layers.push_back({input_dim, 64, 3, 1, 1, 1, true});
    config.cnn_layers.push_back({64, 128, 3, 1, 1, 1, true});
    
    // TDNN layers
    config.tdnn_layers.push_back({128, 256, {-1, 0, 1}, true, 0.1f});
    config.tdnn_layers.push_back({256, 256, {-1, 0, 1}, true, 0.1f});
    config.tdnn_layers.push_back({256, 256, {-3, 0, 3}, true, 0.1f});
    config.tdnn_layers.push_back({256, 256, {-3, 0, 3}, true, 0.1f});
    config.tdnn_layers.push_back({256, 256, {-6, -3, 0, 3, 6}, true, 0.1f});
    
    return config;
}

// Factory for x-vector style config
inline CNNTDNNConfig make_xvector_config(int input_dim, int num_speakers) {
    CNNTDNNConfig config;
    config.input_dim = input_dim;
    config.output_dim = num_speakers;
    
    // No CNN for standard x-vector
    
    // TDNN layers
    config.tdnn_layers.push_back({input_dim, 512, {-2, -1, 0, 1, 2}, true, 0.0f});
    config.tdnn_layers.push_back({512, 512, {-2, 0, 2}, true, 0.0f});
    config.tdnn_layers.push_back({512, 512, {-3, 0, 3}, true, 0.0f});
    config.tdnn_layers.push_back({512, 512, {0}, true, 0.0f});
    config.tdnn_layers.push_back({512, 1500, {0}, true, 0.0f});
    
    return config;
}

}  // namespace cnn
}  // namespace kaldi_fp16

#endif  // __cplusplus
