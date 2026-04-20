#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 12800

// Matrix multiplication function
void matmul_mpi(double *A, double *B, double *C, int rows) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < N; j++) {
            C[i*N + j] = 0.0;
            for (int k = 0; k < N; k++) {
                C[i*N + j] += A[i*N + k] * B[k*N + j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int rows;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Ensure N divisible by number of processes
    if (N % size != 0) {
        if (rank == 0)
            printf("Error: N must be divisible by number of processes\n");
        MPI_Finalize();
        return 0;
    }

    rows = N / size;

    // Host matrices
    double *A = NULL;
    double *B = (double*) malloc(N * N * sizeof(double));
    double *C = NULL;

    // Local matrices
    double *local_A = (double*) malloc(rows * N * sizeof(double));
    double *local_C = (double*) malloc(rows * N * sizeof(double));

    // Initialize only on root
    if (rank == 0) {
        A = (double*) malloc(N * N * sizeof(double));
        C = (double*) malloc(N * N * sizeof(double));

        for (int i = 0; i < N * N; i++) {
            A[i] = 1.0;
            B[i] = 1.0;
        }
    }

    // Broadcast matrix B to all processes
    MPI_Bcast(B, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatter rows of A
    MPI_Scatter(A, rows*N, MPI_DOUBLE,
                local_A, rows*N, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Start timing
    double start = MPI_Wtime();

    // Local computation
    matmul_mpi(local_A, B, local_C, rows);

    // End timing
    double end = MPI_Wtime();

    // Gather results
    MPI_Gather(local_C, rows*N, MPI_DOUBLE,
               C, rows*N, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    // Print result
    if (rank == 0) {
        printf("MPI Time: %f sec\n", end - start);
    }

    // Free memory
    free(B);
    free(local_A);
    free(local_C);

    if (rank == 0) {
        free(A);
        free(C);
    }

    MPI_Finalize();
    return 0;
}
