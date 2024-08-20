#include "common.cuh"
#include "kernel1.cuh"
#include "kernel2.cuh"
#include "kernel3.cuh"
#include "kernel4.cuh"
#include "kernel5.cuh"
#include "kernel6.cuh"
#include "kernel7.cuh"

std::mt19937_64 gen;

void cudaCheck(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s(line %d):\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    return;
};

void CudaDeviceInfo() {
    int deviceId;

    cudaGetDevice(&deviceId);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);

    /*
   * There should be no need to modify the output string below.
   */

    printf("Device ID: %d\n\
       *Number of SMs: %d\n\
       Compute Capability Major: %d\n\
       Compute Capability Minor: %d\n\
       memoryBusWidth: %d\n\
       *maxThreadsPerBlock: %d\n\
       maxThreadsPerMultiProcessor: %d\n\
       *totalGlobalMem: %zuM\n\
       sharedMemPerBlock: %zuKB\n\
       *sharedMemPerMultiprocessor: %zuKB\n\
       totalConstMem: %zuKB\n\
       *multiProcessorCount: %d\n\
       *Warp Size: %d\n",
           deviceId,
           props.multiProcessorCount,
           props.major,
           props.minor,
           props.memoryBusWidth,
           props.maxThreadsPerBlock,
           props.maxThreadsPerMultiProcessor,
           props.totalGlobalMem / 1024 / 1024,
           props.sharedMemPerBlock / 1024,
           props.sharedMemPerMultiprocessor / 1024,
           props.totalConstMem / 1024,
           props.multiProcessorCount,
           props.warpSize);
};

void random_matrix(float* mat, int64_t n)
{
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (auto i = 0; i < n; i++)
    {
            mat[i] = dis(gen);
            //mat[i] = 1;
            //mat[i] = i; 
    }
}   

void copy_matrix(float* src, float* des, int64_t n)
{
    for (int64_t i = 0; i < n; i++)
    {
        des[i] = src[i];
    }
}

bool verify_matrix(float *mat1, float *mat2, int N) 
{
    float diff = 0.0;
    int64_t i;
    for (i = 0; mat1 + i && mat2 + i && i < N; i++) {
        diff = fabs((float) mat1[i] - (float) mat2[i]);
        if (diff > 1e-2) {
            printf("error. %5.2f,%5.2f,%d\n", mat1[i], mat2[i], i);
            return false;
        }
    }
    return true;
}

void matrixdisplay(float *mat, int m, int n)
{
    for (auto i= 0; i < m; i++)
    {
        for (auto j = 0; j < n; j++)
        {
            printf("%f, ", mat[i*n+j]);
        }
        printf("\n");
    }
}

#define CEIL_DIV(M, N) ((M-1)/N + 1)

void test_cublas(cublasHandle_t handle, int M, int N, int K, float alpha, float* A, float* B, float beta, float* C)
{
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N); // float percision
}

void test_mySgemm_v1(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C)
{
    dim3 blockDim(32, 32);
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    mySgemm_v1<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void test_mySgemm_v2(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C)
{
    dim3 blockDim(32, 32);
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    mySgemm_v2<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void test_mySgemm_v3(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C)
{
    const int BS = 32;
    dim3 blockDim(32, 32);
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    mySgemm_v3<BS><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
} 

void test_mySgemm_v4(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C)
{
    const int BM = 64;
    const int BN = 64;
    const int BK = 16;
    const int TM = 8;

    assert(BM*BK >= BM * BN/ (TM ));
    assert(BK*BN >= BM * BN/ (TM ));
    dim3 blockDim(BM*BN/TM);
    dim3 gridDim(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    mySgemm_v4<BM, BN, BK, TM><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
} 

void test_mySgemm_v5(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C)
{
    const int BM = 128;
    const int BN = 128;
    const int BK = 32;
    const int TM = 8;
    const int TN = 8;

    assert(BM*BK >= BM * BN/ (TM * TN));
    assert(BK*BN >= BM * BN/ (TM * TN));

    dim3 blockDim(BM*BN/(TM*TN));
    dim3 gridDim(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    mySgemm_v5<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void test_mySgemm_v6(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C)
{
    const int BM = 128;
    const int BN = 128;
    const int BK = 16;
    const int TM = 8;
    const int TN = 8;

    assert(BM*BK >= 4 * BM * BN/ (TM * TN));
    assert(BK*BN >= 4 * BM * BN/ (TM * TN));

    dim3 blockDim(BM*BN/(TM*TN));
    dim3 gridDim(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    mySgemm_v6<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void test_mySgemm_v7(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C)
{
    const int BM = 128;
    const int BN = 64;
    const int BK = 8;
    const int WM = 64;
    const int WN = 32;
    const int WNITER = 2;
    const int TM = 4;
    const int TN = 4;

    const int WMITER = (WM * WN) / (32 * TM * TN * WNITER);
    const int thread_num = BM * BN / (TM * TN * WMITER * WNITER);
    const int warp_num_in_block = thread_num / 32;

    //const int BM = 128;
    //const int BN = 64;
    //const int BK = 16;
    //const int WM = 32;
    //const int WN = 16;
    //const int WNITER = 1;
    //const int TM = 4;
    //const int TN = 4;

    assert(BM * BK >= 4 * thread_num);
    assert(BK * BN >= 4 * thread_num);
    //assert(warp_num_in_block == BM * BN / (WM * WN)); // aseert the number of warps equals to the number of warp tiles;

    dim3 blockDim(thread_num);
    dim3 gridDim(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    mySgemm_v7<BM, BN, BK, WM, WN, WNITER, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void test_kernel(int kernel_num, int M, int N, int K, float alpha, float* A, float* B, float beta, float* C, cublasHandle_t handle)
{
    switch(kernel_num)
    {
        case 0:
            test_cublas(handle, M, N, K, alpha, A, B, beta, C);
            break;
        case 1:
            test_mySgemm_v1(M, N, K, alpha, A, B, beta, C);
            break;
        case 2:
            test_mySgemm_v2(M, N, K, alpha, A, B, beta, C);
            break;
        case 3:
            test_mySgemm_v3(M, N, K, alpha, A, B, beta, C);
            break;
        case 4:
            test_mySgemm_v4(M, N, K, alpha, A, B, beta, C);
            break;
        case 5:
            test_mySgemm_v5(M, N, K, alpha, A, B, beta, C);
            break;
        case 6:
            test_mySgemm_v6(M, N, K, alpha, A, B, beta, C);
            break;
        case 7:
            test_mySgemm_v7(M, N, K, alpha, A, B, beta, C);
            break;
        default:
            break;
    }
}
