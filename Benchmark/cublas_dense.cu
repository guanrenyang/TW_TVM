#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i)) // macro to index a 2D array stored in row-major order

#include <cublas_v2.h>

#include <cublas_v2.h>

void matrixMul(const float *A, const float *B, float *C, const int m, const int n, const int k) {
    // Allocate device memory for matrices A, B, and C
    float *d_A, *d_B, *d_C;
    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc((void**)&d_A, m*k*sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        return;
    }
    cudaStatus = cudaMalloc((void**)&d_B, k*n*sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_A);
        return;
    }
    cudaStatus = cudaMalloc((void**)&d_C, m*n*sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_A);
        cudaFree(d_B);
        return;
    }

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&handle);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS initialization failed\n");
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    // Copy matrices A and B from host to device
    cublasStatus = cublasSetMatrix(m, k, sizeof(float), A, m, d_A, m);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS set matrix A failed\n");
        cublasDestroy(handle);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }
    cublasStatus = cublasSetMatrix(k, n, sizeof(float), B, k, d_B, k);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS set matrix B failed\n");
        cublasDestroy(handle);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    // Allocate CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start);

    // Compute C = alpha*A*B + beta*C using cuBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS matrix multiplication failed\n");
        cublasDestroy(handle);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    // Record stop event and wait for completion
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Elapsed time: %f ms\n", elapsedTime);
 
    // Copy matrix C from device to host
    cublasStatus = cublasGetMatrix(m, n, sizeof(float), d_C, m, C, m);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS get matrix C failed\n");
        cublasDestroy(handle);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Destroy cuBLAS handle
    cublasDestroy(handle);

}


void matrixMul_cpu(const float *A, const float *B, float *C, const int m, const int n, const int k) {
    // Perform matrix multiplication C = A*B
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < k; ++l) {
                sum += A[i*k+l] * B[l*n+j];
            }
            C[i*n+j] = sum;
        }
    }
}

int main() {
    const int m = 1024; // number of rows in matrix A and matrix C
    const int n = 1024; // number of columns in matrix B and matrix C
    const int k = 1024; // number of columns in matrix A and number of rows in matrix B

    // Allocate host memory for matrices A, B, and C
    float *h_A = (float*)malloc(m*k*sizeof(float));
    float *h_B = (float*)malloc(k*n*sizeof(float));
    float *h_C = (float*)malloc(m*n*sizeof(float));
    float *h_C_verify = (float*)malloc(m*n*sizeof(float));

    // Initialize matrices A and B with random values
    for (int i = 0; i < m*k; ++i) {
        h_A[i] = ((float) rand()) / ((float) RAND_MAX);
    }
    for (int i = 0; i < k*n; ++i) {
        h_B[i] = ((float) rand()) / ((float) RAND_MAX);
    }

    // Compute matrix C = A*B using GPU kernel
    matrixMul(h_A, h_B, h_C, m, n, k);
    matrixMul(h_A, h_B, h_C, m, n, k);
    matrixMul(h_A, h_B, h_C, m, n, k);
    matrixMul(h_A, h_B, h_C, m, n, k);
    // matrixMul_cpu(h_A, h_B, h_C_verify, m, n, k);

    // // Print matrix C
    // printf("Matrix C =\n");
    // for (int i = 0; i < m; ++i) {
    //     for (int j = 0; j < n; ++j) {
    //         printf("%f ", h_C[IDX2C(i,j,m)]);
    //     }
    //     printf("\n");
    // }



    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_verify);
    return 0;
}

