# Examples CMakeLists.txt

# Basic matrix multiplication example
add_executable(basic_matmul basic_matmul.cu)
target_link_libraries(basic_matmul kaldi-fp16)

# Performance benchmark
add_executable(benchmark_fp16 benchmark_fp16.cu)
target_link_libraries(benchmark_fp16 kaldi-fp16)

# Precision comparison
add_executable(compare_precision compare_precision.cu)
target_link_libraries(compare_precision kaldi-fp16)

# Install examples
install(TARGETS basic_matmul benchmark_fp16 compare_precision
    RUNTIME DESTINATION bin/examples
)
