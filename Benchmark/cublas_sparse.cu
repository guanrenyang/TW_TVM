#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string>
#include <fstream>
float matMulSp(const int M, const int K, const int N, const float sparsity)
{

    // Allocate memory for the sparse matrix A and dense matrix B
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));

    // Initialize the sparse matrix A with random values
    for (int i = 0; i < M * K; i++)
    {
        if ((float)rand() / RAND_MAX < sparsity)
            h_A[i] = (float)rand() / RAND_MAX;
        else
            h_A[i] = 0.0f;
    }

    // Initialize the dense matrix B with random values
    for (int i = 0; i < K * N; i++)
    {
        h_B[i] = (float)rand() / RAND_MAX;
    }

    // Allocate memory for the result matrix C
    float *h_C = (float*)malloc(M * N * sizeof(float));

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Allocate memory on the device for the sparse matrix A and dense matrix B
    float *d_A, *d_B;
    cudaMalloc((void**)&d_A, M * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * N * sizeof(float));

    // Copy the sparse matrix A and dense matrix B from host to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define the alpha and beta coefficients
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Create the cuBLAS descriptor
    cublasSpMatDescr_t matA;
    cublasCreateSpMat(&matA, M, K, M * K, d_A, CUBLAS_DATA_TYPE_FLOAT, CUBLAS_INDEX_TYPE_INT32, CUBLAS_SPARSE_CSR);

    // Create the cuBLAS operation descriptor
    cublasDnMatDescr_t matB;
    cublasCreateDnMat(&matB, K, N, K, d_B, CUBLAS_DATA_TYPE_FLOAT, CUBLAS_ORDER_COL);

    // Allocate memory on the device for the result matrix C
    float *d_C;
    cudaMalloc((void**)&d_C, M * N * sizeof(float));

    // Allocate CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start);

    // Perform the sparse matrix multiplication using cuBLAS
    cublasSpMatDescr_t matC;
    cublasCreateSpMat(&matC, M, N, M * N, d_C, CUBLAS_DATA_TYPE_FLOAT, CUBLAS_INDEX_TYPE_INT32, CUBLAS_SPARSE_CSR);
    cublasSpMM(handle, CUBLAS_OP_N, CUBLAS_OP_N, &alpha, matA, matB, &beta, matC, CUBLAS_DATA_TYPE_FLOAT, CUBLAS_SPMM_ALG_DEFAULT);

    // Record stop event and wait for completion
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Elapsed time: %f ms\n", elapsedTime);

    // Copy the result matrix C from device to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print matrix C
    printf("Matrix C =\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            printf("%f ", h_C[i * M + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return elapsedTime;
}

int main(){
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;
    string log_file = 'log_cublas_sparse';
    std::ofstream outfile;
    outfile.open(log_file);
    for (float spar = 0.01; spar<1.0; spar+=0.01){
        float elapsedTime = matMulSp(M, N, K, spar);
        outfile << std::fixed << std::setprecision(3) << spar << " " << std::fixed << std::setprecision(3) << elapsedTime << std::endl;
    }
    outfile.close();
    return 0;
}