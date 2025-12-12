#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void serial_matrix_mult(int* h_a, int* h_b, int* h_result, int n);

int main() {
    int n = 1280;
    int *a, *b, *c;

    // Allocate memory
    a = (int*)malloc(sizeof(int) * n * n);
    b = (int*)malloc(sizeof(int) * n * n);
    c = (int*)malloc(sizeof(int) * n * n);

    
    // Initialize matrix A with 2
    for (int i = 0; i < n * n; i++) {
        a[i] = 2;
    }

    // Initialize matrix B with 3
    for (int i = 0; i < n * n; i++) {
        b[i] = 3;
    }

    // Measure execution time
    clock_t start = clock();
    serial_matrix_mult(a, b, c, n);
    clock_t stop = clock();

    double timeTakenMs = ((double)(stop - start) / CLOCKS_PER_SEC) * 1000.0;

    printf("Matrix size: %d x %d\n", n, n);
    printf("Time taken: %f ms\n", timeTakenMs);
    printf("Sample result c[12][0] = %d\n", c[12 * n + 0]);

    // Free memory
    free(a);
    free(b);
    free(c);

    return 0;
}

void serial_matrix_mult(int* h_a, int* h_b, int* h_result, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int tmp = 0;
            for (int k = 0; k < n; ++k) {
                tmp += h_a[i * n + k] * h_b[k * n + j];
            }
            h_result[i * n + j] = tmp;
        }
    }
}

