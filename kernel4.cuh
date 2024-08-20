#include <cassert>

template<const int BM, const int BN, const int BK, const int TM>
__global__ void mySgemm_v4(int m, int n, int k, float alpha, float *A, float *B, float beta, float *C)
{
    // 1-d block 
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int thread_num = BM * BN / TM; // one thread in a block will calculate TM elements

    assert(thread_num == blockDim.x); // assert the number of threads equal to BM * BN / TM

    int tx = threadIdx.x % BN;
    int ty = threadIdx.x / BN * TM;

    // allocate share memory
    __shared__ float As[BM*BK];
    __shared__ float Bs[BK*BN];

    // move to current A, B and C address
    A = &A[by * BM * k];
    B = &B[bx * BN];
    C = &C[by * BM * k + bx * BN];


    // calclate the row and col for a_tile and b_tile for shared memory

    //printf("threadIdx.x = %d\n", threadIdx.x);
    unsigned int a_tile_row = threadIdx.x / BK ;
    unsigned int a_tile_col = threadIdx.x % BK;
    unsigned int a_tile_stride = thread_num / BK;

    //assert(TM * a_tile_stride = BM);
    //printf("bx, by = %d, %d, a_tile_row = %d, a_tile_col = %d, a_tile_stride = %d\n", bx, by, a_tile_row, a_tile_col, a_tile_stride);

    unsigned int b_tile_row = threadIdx.x / BN;
    unsigned int b_tile_col = threadIdx.x % BN;
    unsigned int b_tile_stride = thread_num / BN;
    //printf("b_tile_row = %d, b_tile_col = %d, b_tile_stride = %d\n", b_tile_row, b_tile_col, b_tile_stride);

    float tmp[TM] = {0.};
    for (auto k0 = 0; k0 < k; k0+=BK)
    {
        // copy element A[a_tilte_row][a_title_col] from DRAM to shared MEM As[a_title_row][a_tilte_col] 
        #pragma unroll 
        for (auto i = 0; i < BM; i+=a_tile_stride)
        {
            As[(a_tile_row + i) * BK + a_tile_col] = A[(a_tile_row + i) * k + a_tile_col];
            //if (bx == 0 && by == 0)
            //    printf("As[%d] = A[%d], %f = %f\n", (a_tile_row + i) * BK + a_tile_col, (a_tile_row + i) * k + a_tile_col, As[(a_tile_row + i) * BK + a_tile_col], A[(a_tile_row + i) * k + a_tile_col]);
        }

        #pragma unroll 
        for (auto j = 0; j < BK; j += b_tile_stride)
        {
            Bs[(b_tile_row + j) * BN + b_tile_col] = B[(b_tile_row + j) * n + b_tile_col];
        }

        __syncthreads();

        //displayshared(&As, BM, BK);
        A += BK;
        B += BK * n;

        for (auto i = 0; i < BK; i++)
        {
            auto tmpb = Bs[i*BN + tx];
            for (auto l = 0; l < TM; l++)
            {
                tmp[l] += As[(ty + l) * BK + i] * tmpb;      
                //if (ty == 0 && tx == 0 && bx == 0 && by == 0 && i == 0 && l == 0)
                //    printf(" tmp[%d] = %f * %f\n", l, As[(ty + l) * BK + i], tmpb); 
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (auto i = 0; i < TM; i++)
    {
        C[(ty + i) * n + tx] = alpha * tmp[i] + beta * C[(ty + i) * n + tx] ;
    }
}

