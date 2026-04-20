#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 10000

void matmul_mpi(double *A, double *B, double *C, int rows) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < N; j++) {
            C[i*N + j] = 0.0;
            for (int k = 0; k < N; k++)
                C[i*N + j] += A[i*N + k] * B[k*N + j];
        }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int rows;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    rows = N / size;

    double *A = NULL;
    double B = (double) malloc(N*N*sizeof(double));
    double *C = NULL;
    double local_A = (double) malloc(rows*N*sizeof(double));
    double local_C = (double) malloc(rows*N*sizeof(double));

    if (rank == 0) {
        A = (double*) malloc(N*N*sizeof(double));
        C = (double*) malloc(N*N*sizeof(double));

        for (int i = 0; i < N*N; i++) {
            A[i] = 1.0;
            B[i] = 1.0;
        }
    }

    MPI_Bcast(B, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatter(A, rows*N, MPI_DOUBLE,
                local_A, rows*N, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    double start = MPI_Wtime();

    matmul_mpi(local_A, B, local_C, rows);

    double end = MPI_Wtime();

    MPI_Gather(local_C, rows*N, MPI_DOUBLE,
               C, rows*N, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (rank == 0)
        printf("MPI Time: %f sec\n", end - start);

    MPI_Finalize();
    return 0;
}
