#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define BLOCK_SIZE 16

// CUDA Kernel (Based on GPU-GEMM powerpoint)
__global__ void matMultKernel(float* a, float* b, float* c, int n, int m){
        int x = threadIdx.x;
        int i = blockIdx.x * BLOCK_SIZE + x;
        int j = blockIdx.y * 2;
        __shared__ float sub1[BLOCK_SIZE];
        __shared__ float sub2[BLOCK_SIZE];
        float sum0 = 0.0f;
        float sum1 = 0.0f;

        for(int y = 0; y < m; y += BLOCK_SIZE){
                sub1[x] = b[y + x + m * j];
                sub2[x] = b[y + x + m * (j + 1)];
                __syncthreads();

                for(int z = y; z < y+BLOCK_SIZE; ++z){
                        float r = a[i + n * z];
                        sum0 += r * sub1[z - y];
                        sum1 += r * sub2[z - y];
                }
                __syncthreads();
        }

        c[i + n * j] = sum0;
        c[i + n * (j + 1)] = sum1;
}

int main(int argc, char **argv) {
        FILE *mat1FP;
        FILE *mat2FP;
        float *A, *B, *C1, *C2;
        unsigned int mat1dims[2]; // 0 is rows 1 is columns
        unsigned int mat2dims[2]; // 0 is rows 1 is columns
        long long numA, numB, numC, numOps; // Element counts for each matrix
        cudaEvent_t start1, start2, stop1, stop2;
        mat1FP = fopen(argv[1], "r");
        mat2FP = fopen(argv[2], "r");
        fread(mat1dims, 4, 2, mat1FP);
        fread(mat2dims, 4, 2, mat2FP);

        numA = mat1dims[0]*mat1dims[1];
        numB = mat2dims[0]*mat2dims[1];
        numC = mat1dims[0]*mat2dims[1];
        numOps = (long long)mat1dims[0] * (long long)mat1dims[1] * (long long)mat2dims[1];

        A = (float *)malloc(mat1dims[0]*mat1dims[1]*sizeof(float));
        B = (float *)malloc(mat2dims[0]*mat2dims[1]*sizeof(float));
        C1 = (float *)malloc(numC*sizeof(float));
        C2 = (float *)malloc(numC*sizeof(float));

        // The matrices are stored in column-order
        fread(A, 4, mat1dims[0]*mat1dims[1], mat1FP);
        fread(B, 4, mat2dims[0]*mat2dims[1], mat2FP);

        // Initialize cuBLAS
        cublasHandle_t handle;
        cudaEventCreate(&start1);
        cudaEventCreate(&stop1);
        cublasCreate(&handle);
        float *gpu_A, *gpu_B, *gpu_C;

        // Allocate device memory and matrices
        cudaMalloc(&gpu_A, numA * sizeof(float));
        cudaMemcpy(gpu_A, A, numA * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&gpu_B, numB * sizeof(float));
        cudaMemcpy(gpu_B, B, numB * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&gpu_C, numC * sizeof(float));

        float alpha = 1.0;
        float beta = 1.0;
        int M, N, K;
        M = mat1dims[0]; N = mat2dims[1]; K = mat1dims[1];

        cudaEventRecord(start1, 0);
        cudaEventSynchronize(start1);
        cublasSgemm(    handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        M, N, K,
                        &alpha, gpu_A, M,
                        gpu_B, K, &beta,
                        gpu_C, M);
        cudaDeviceSynchronize();

        cudaEventRecord(stop1, 0);
        cudaEventSynchronize(stop1);
        float elapsedTime_1 = 0;
        double GFLOPS_1 = 0;
        cudaEventElapsedTime(&elapsedTime_1, start1, stop1);
        GFLOPS_1 = (((2000*(double)numOps) / elapsedTime_1) / 1000000000.0);
        printf("\ncuBLAS SGEMM runtime: %f milliseconds.\n", elapsedTime_1);
        printf("cuBLAS SGEMM performance: %f GFLOPS\n", GFLOPS_1);

        cudaMemcpy(C1, gpu_C, numC*sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(gpu_A);
        cudaFree(gpu_B);
        cudaFree(gpu_C);
        // End of cuBLAS SGEMM implementation

        // Shared memory implementation
        dim3 threads(BLOCK_SIZE);
        dim3 grid(mat1dims[0]/BLOCK_SIZE, mat2dims[1]/2);

        cudaMalloc(&gpu_C, numC * sizeof(float));
        cudaMalloc(&gpu_A, numA * sizeof(float));
        cudaMemcpy(gpu_A, A, numA * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&gpu_B, numB * sizeof(float));
        cudaMemcpy(gpu_B, B, numB * sizeof(float), cudaMemcpyHostToDevice);

        cudaEventCreate(&start2);
        cudaEventCreate(&stop2);
        cudaEventRecord(start2, 0);
        cudaEventSynchronize(start2);

        matMultKernel<<<grid, threads>>>(gpu_A, gpu_B, gpu_C, mat1dims[0], mat2dims[0]);

        cudaDeviceSynchronize();
        cudaEventRecord(stop2, 0);
        cudaEventSynchronize(stop2);

        float elapsedTime_2 = 0;
        double GFLOPS_2 = 0;
        cudaEventElapsedTime(&elapsedTime_2, start2, stop2);
        GFLOPS_2 = (((2000*(double)numOps) / elapsedTime_2) / 1000000000.0);
        printf("\nShared Memory runtime: %f milliseconds.\n", elapsedTime_2);
        printf("Shared Memory performance: %f GFLOPS\n", GFLOPS_2);

        cudaMemcpy(C2, gpu_C, numC*sizeof(float), cudaMemcpyDeviceToHost);

        float maxDiff = 0;
        for (int i = 0; i < numC; i++) {
                if (fabs(C1[i] - C2[i]) > 0) {
                        maxDiff = fabs(C1[i] - C2[i]);
                }
        }
        printf("\n MAX DIFF: %f\n", maxDiff);

        FILE *outFile = fopen(argv[3], "w");
        printf("Writing result matrix to %s\n", argv[3]);
        int outMatDims[2] = { mat1dims[0], mat2dims[1] };
        fwrite(outMatDims, sizeof(int), 2, outFile);
        fwrite(C1, sizeof(float), outMatDims[0]*outMatDims[1], outFile);

        free(A);
        free(B);
        free(C1);
        free(C2);
        cudaFree(gpu_A);
        cudaFree(gpu_B);
        cudaFree(gpu_C);
        cublasDestroy(handle);
        return 0;
}
