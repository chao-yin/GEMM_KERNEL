template<const int BS> 
__global__ void mySgemm_v3(int m, int n, int k, float alpha, float *A, float *B, float beta, float *C)
{

    __shared__ float As[BS*BS];
    __shared__ float Bs[BS*BS];

    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0f;

    const int innerRow = threadIdx.y;
    const int innerCol = threadIdx.x;
    if (row < m && col < n)
    {   
        for(auto tile = 0; tile < k; tile+=BS)
        {   
            // copy A and B tile from DRAM to shared memory
            As[innerRow*BS+innerCol] = A[row*k+tile+innerCol];
            Bs[innerRow*BS+innerCol] = B[(tile+innerRow)*n+innerCol];

            __syncthreads(); // threads wait for each other to finish loading before computing

            for (auto i = 0; i < BS; i++)
            {   
                sum += alpha * As[innerRow*BS+i] * Bs[i*BS+innerCol];
            }   

            __syncthreads(); // threads wait for each other to fish computing before loading
        }   
            
        C[row*n+col] = sum + beta * C[row*n+col];
    }   

}

