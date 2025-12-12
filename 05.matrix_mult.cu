#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define MATRIX_SIZE 1280

// CUDA kernel for matrix multiplication using global memory
__global__ void matrixMultiply(float *A, float *B, float *C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        float sum = 0.0;
        for (int k = 0; k < width; ++k) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

int main() {
    int width = MATRIX_SIZE;
    size_t size = width * width * sizeof(float);

    // Host matrices and result
    float *h_A, *h_B, *h_C;
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);

    // Initialize matrices A and B
    for (int i = 0; i < width * width; ++i) {
        h_A[i] = 1.0; // Replace with your initialization
        h_B[i] = 2.0; // Replace with your initialization
    }

    // Device matrices and result
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define block size
    int BLOCK_SIZE = 16;

    // Define grid and block dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (width + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel for matrix multiplication
    matrixMultiply<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, width);

    // Copy matrix C from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify results (print some elements if needed)
    printf("Sample result: C[0][0] = %f\n", h_C[0]);

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
