template<const int BM,
         const int BN,
         const int BK,
         const int TM,
         const int TN>
__global__ void mySgemm_v5(int m, int n, int k, float alpha, float *A, float *B, float beta, float *C)
{
    int bx = blockIdx.y;
    int by = blockIdx.x;

    const int threads_num = (BM * BN) / (TM * TN);

    assert(threads_num == blockDim.x); 

    int block_row_thread = BM / TM;
    int block_col_thread = BN / TN;

    int tx = (threadIdx.x % block_col_thread) * TN;
    int ty = (threadIdx.x / block_col_thread) * TM;

    int a_tile_row = threadIdx.x / BK;
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride = threads_num / BK;

    int b_tile_row = threadIdx.x / BM;
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = threads_num / BN;

    // alloc shared meemort As and Bs
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];


    // move to current A, B and C address
    A = &A[by * BM * k];
    B = &B[bx * BN];
    C = &C[by * BM * k + bx * BN];

    float tmp[TM * TN] = {0.};
    float A_reg[TM] = {0.};
    float B_reg[TN] = {0.};
    for (int k0 = 0; k0 < k; k0+=BK)
    {
        // copy A and B from DRAM to As and Bs
        #pragma unroll
        for (auto i = 0; i < BM; i+=a_tile_stride)
        {
            As[(a_tile_row+i) * BK + a_tile_col] = A[(a_tile_row+i) * k + a_tile_col];
            //if (bx == 0 && by == 0)
            //    printf("bx, by = %d, %d,  k0 = %d,  i = %d,  As[%d] = A[%d]; %f = %f\n", bx, by, k0, i, (a_tile_row + i) * BK + a_tile_col, (a_tile_row + i) * k + a_tile_col,  As[(a_tile_row + i) * BK + a_tile_col], A[(a_tile_row + i) * k + a_tile_col]);
        }

        #pragma unroll
        for (auto i = 0; i < BK; i+=b_tile_stride)
        {
            Bs[(b_tile_row+i) * BN + b_tile_col] = B[(b_tile_row+i) * n + b_tile_col];
            //if (bx == 0 && by == 0)
            //    printf("Bs[%d][%d] = B[%d][%d],  %f = %f\n", (b_tile_row+i),  b_tile_col, (a_tile_row+i), b_tile_col, Bs[(b_tile_row+i) * BN + b_tile_col] , B[(a_tile_row+i) * n + b_tile_col]);
        }

        __syncthreads();

        // update A and B
        A += BK;
        B += BK * n;
        #pragma unroll
        for (auto p = 0; p < BK; p++)
        {
            // something like packing
            for (auto i0 = 0 ; i0 < TM; i0++)
            {
                A_reg[i0] = As[(ty + i0) * BK + p];
            }

            for (auto j0 = 0; j0 < TN; j0++)
            {
                B_reg[j0] = Bs[p * BN + tx + j0];
            }

            #pragma unroll
            for (auto i = 0; i < TM; i++)
            {
                float tmpa = A_reg[i];
                for (auto j = 0; j < TN; j++)
                {
                    //tmp[i*TN+j] += tmpa * Bs[p * BN + tx + j];
                    tmp[i * TN + j] += tmpa * B_reg[j];
                    //if (ty == 0 && tx == 0 && bx == 0 && by == 0 && i == 0 && j == 0)
                    //    printf(" tmp[%d] = %f * %f\n", i * TN + j, tmpa, B_reg[j]); 
                } 
            }       
        }
        __syncthreads();
    }
    #pragma unroll
    for (auto i = 0; i < TM; i++)
    for (auto j = 0; j < TN; j++)
    {
        C[(ty + i) * n + tx + j] = alpha * tmp[i * TN + j] + beta * C[(ty + i) * n + tx + j] ;
        //if (ty == 0 && tx == 0 && bx == 0 && by == 0 && i == 0 && j == 0)
        //    printf(" C[(ty + i) * n + tx + j] = %f\n ",  C[(ty + i) * n + tx + j]);
    }
}
