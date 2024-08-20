#ifndef __COMMON_CUH__
#define __COMMON_CUH__

#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>


/*
**********************

CUDA operations

**********************
*/

void CudaDeviceInfo();                // print device properity 
void cudaCheck(cudaError_t error, const char *file, int line); 

/*
*********************

Matrix operations

*********************
*/ 

void random_matrix(float* mat, int64_t N);            // initalize matrix randomly
void copy_matrix(float *src, float *dest, int64_t N);    // copy matrix
bool verify_matrix(float* mat1, float* mat2, int N);    // verify the correctness of matrix multiplication
void matrixdisplay(float *mat, int m, int n);

/*
*********************

KERNEL testing function

*********************
*/


void test_kernel(int kernel_num, int m, int n, int k, float alpha, float *A, float *B, float beta, float *C, cublasHandle_t handle);
#endif
