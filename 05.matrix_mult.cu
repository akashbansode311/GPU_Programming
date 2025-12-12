#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define MATRIX_SIZE 1280  // Define the matrix size

// GPU kernel
__global__ void gpu_matrix_mult(int *a, int *b, int *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;

    if (row < n && col < n) {
        for (int i = 0; i < n; i++)
            sum += a[row * n + i] * b[i * n + col];
        c[row * n + col] = sum;
    }
}

int main() {
    int n = MATRIX_SIZE;  // Use the macro for matrix size
    printf("Matrix size chosen: %d x %d\n", n, n);

    // Host (pinned) memory
    int *h_a = (int*)malloc(sizeof(int) * n * n);
    int *h_b = (int*)malloc(sizeof(int) * n * n);
    int *h_c = (int*)malloc(sizeof(int) * n * n);


    // Initialize matrices
    for (int i = 0; i < n * n; i++) {
        h_a[i] = 2;
        h_b[i] = 3;
    }

    // Device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, sizeof(int) * n * n);
    cudaMalloc((void **)&d_b, sizeof(int) * n * n);
    cudaMalloc((void **)&d_c, sizeof(int) * n * n);

    // Copy data to device
    cudaMemcpy(d_a, h_a, sizeof(int) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int) * n * n, cudaMemcpyHostToDevice);

    // Define block size before dim3
    int BLOCK_SIZE = 16;  // Threads per block in x and y

    // Compute grid and block sizes
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Copy result back
    cudaMemcpy(h_c, d_c, sizeof(int) * n * n, cudaMemcpyDeviceToHost);

    // Measure elapsed time
    float gpu_time_ms;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);

    printf("Naive GPU matrix multiplication %dx%d: %f ms\n", n, n, gpu_time_ms);
    printf("Sample result C[12][0] = %d (expected %d)\n", h_c[12 * n + 0], 2*3*n);

    // Free memory
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
