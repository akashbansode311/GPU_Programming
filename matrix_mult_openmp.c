#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 10000

void matmul_omp(double A[N][N], double B[N][N], double C[N][N]) {

    #pragma omp parallel for
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
}

int main() {
    static double A[N][N], B[N][N], C[N][N];
    double start, end;

    // Init
    #pragma omp parallel for
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = 1.0;
            B[i][j] = 1.0;
        }

    start = omp_get_wtime();
    matmul_omp(A, B, C);
    end = omp_get_wtime();

    printf("OpenMP Time: %f sec\n", end - start);

    return 0;
}
