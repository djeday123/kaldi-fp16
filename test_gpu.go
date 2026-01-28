package main

import (
    "fmt"
    "kaldi-fp16/go/kaldibridge"
)

func main() {
    // Создаём cuBLAS handle
    handle, err := kaldibridge.NewCuBLASHandle()
    if err != nil {
        fmt.Println("Error:", err)
        return
    }
    defer handle.Close()
    
    // Создаём тензоры
    A, _ := kaldibridge.NewTensorGPU(1024, 512)
    B, _ := kaldibridge.NewTensorGPU(512, 256)
    
    // Матричное умножение с Tensor Cores
    C, err := kaldibridge.MatMulGPU(handle, A, B)
    if err != nil {
        fmt.Println("MatMul error:", err)
        return
    }
    
    fmt.Printf("✓ GPU Tensor Cores работают!\n")
    fmt.Printf("  A: %dx%d\n", A.Rows(), A.Cols())
    fmt.Printf("  B: %dx%d\n", B.Rows(), B.Cols())
    fmt.Printf("  C: %dx%d\n", C.Rows(), C.Cols())
    
    A.Free()
    B.Free()
    C.Free()
}