#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "common.cuh"

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        printf("Plese select a kernel (between 0 - 10, here 0 is Nivida cuBlas) \n");
        exit(EXIT_FAILURE);
    }

    // cuda kernel num
    int kernel_num = atoi(argv[1]);
    if ( kernel_num < 0 || kernel_num > 10)
    {
        printf("Please enter a valid kernel number (0 - 10)\n");
        exit(EXIT_FAILURE);
    }
    else
    {
        printf("Kenel number is %d\n", kernel_num);
    }

    cublasHandle_t handle;
    if (cublasCreate(&handle))
    {
        printf("Create cublas handle error.\n");
        exit(EXIT_FAILURE);
    }

    // timeing, create cuda event to record the time 

    float elapsed_time;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    // matrix size
    int size_len = 24;
    int SIZE[size_len];
    for ( int i = 0; i < size_len; i++)
        SIZE[i] = 256 * (i + 1);

    int m, n, k, max_size;  
    max_size = SIZE[size_len - 1];
    printf("max_size = %d\n", max_size);

    float alpha = 1.0, beta = 1.0;
    float *A = NULL, *B = NULL, *C = NULL, *C_ref = NULL; // host matrix
    float *dA = NULL, *dB = NULL, *dC = NULL, *dC_ref = NULL; // device matrix

    A = (float *) malloc(sizeof(float) * max_size * max_size);
    B = (float *) malloc(sizeof(float) * max_size * max_size);
    C = (float *) malloc(sizeof(float) * max_size * max_size);
    C_ref = (float *) malloc(sizeof(float) * max_size * max_size);

    random_matrix(A, max_size * max_size);
    random_matrix(B, max_size * max_size);
    random_matrix(C, max_size * max_size);
    copy_matrix(C, C_ref, max_size * max_size);

    cudaCheck(cudaMalloc((void **) &dA, sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc((void **) &dB, sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc((void **) &dC, sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc((void **) &dC_ref, sizeof(float) * max_size * max_size));

    // transfer data from host to device
    cudaMemcpy(dA, A, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice);
    cudaCheck(cudaMemcpy(dB, B, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));

    int repeated_times = 10;
    for (int i = 0; i < size_len; i++)
    {
        cudaCheck(cudaMemcpy(dC, C, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(dC_ref, C_ref, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));
        m = n = k = SIZE[i];

        printf("m=n=k=%d\n", m);

        // verify the correctness of compuation, and excute the kernel before real computation to avoid cold start

        if (kernel_num != 0)
        {
            test_kernel(0, m, n, k, alpha, dA, dB, beta, dC_ref, handle);  // cuBlas
            test_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);  // user define
            cudaDeviceSynchronize();
            cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
            cudaMemcpy(C_ref, dC_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            //printf("\n\n*********C********\n\n");
            //matrixdisplay(C, m, n);

            //printf("\n\n*********C_ref********\n\n");
            //matrixdisplay(C_ref, m, n);

            if (!verify_matrix(C_ref, C, m * n))
            {
                printf("Failed to pass the correctness verification against cuBlas, Exited.\n");
                exit(EXIT_FAILURE);
            }
        }

        cudaDeviceSynchronize();

        cudaEventRecord(beg);
        for ( int i = 0; i < repeated_times; i++)
        {
            test_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);
        }
        cudaEventRecord(end);
        cudaEventSynchronize(beg);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time, beg, end);
        elapsed_time /= 1000.; // convert to second

        printf("Average elasped time: (%f) second, performance: (%f) GFLOPS. size: (%d).\n", 
                elapsed_time / repeated_times, 2 * 1e-9 * repeated_times * m * n * k /  elapsed_time, m);
    
        fflush(stdout);
        copy_matrix(C_ref, C, m * n); //sync C with cuBLAS to prepare for the next run
    }

    free(A);
    free(B);
    free(C);
    free(C_ref);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dC_ref);

    return 0;
}
