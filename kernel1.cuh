__global__ __launch_bounds__(1024) void mySgemm_v1(int m, int n, int k, float alpha, float* A, float* B, float beta, float* C) 
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    const int lda = k;
    const int ldb = n;
    const int ldc = n;
    //printf("idx, idy = %d, %d\n", idx, idy);

    if (idx < m && idy < n)
    {
        float sum = 0.0;
        for(int i = 0; i < k; i++)
        {
            sum += A[idx*lda + i] * B[i*ldb+idy];
        }
        // C[x*ldc+y] = alpha * sum + beta * C[x*ldc + y]
        C[idx*ldc + idy] = alpha * sum + beta * C[idx*ldc+idy];
    }
}

