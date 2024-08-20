
const int warpsize = 32;

template<const int BM,
         const int BN,
         const int BK,
         const int WM,
         const int WN,
         const int WNITER,
         const int TM,
         const int TN>
__global__ void mySgemm_v7(int m, int n, int k, float alpha, float *A, float *B, float beta, float *C)
{
    int by = blockIdx.x;
    int bx = blockIdx.y;

    //const int threads_num = BM * BN / (TM * TN);

    //assert( blockDim.x == threads_num);


    // warp paralization , get the row and col of warp in block
    int warpIdx = threadIdx.x / warpsize; 
    int warpcol = warpIdx % (BN / WN);
    int warprow = warpIdx / (BN / WN);
    // int warp_stride = BN / WN;

    //assert( threads_num / warpsize == BM * BN / (WM * WN)); // aseert the number of warps equals to the number of warp tiles;

    //assert( WM * WN / (WMINTER * WNITER) == warpsize * TM * TN);
    const int WMITER = (WM * WN) / (warpsize * TM * TN * WNITER); // how many loops in row directions , this value should be the same as WNITER

    const int threads_num = BM * BN / (TM * TN * WMITER * WNITER);

    int WSUBM = WM / WMITER;  // warp tile size in row
    int WSUBN = WN / WNITER;  // warp tile size in col


    // get the row and col of threadIdx.x in a warp
    const int threadIdxInWarp = threadIdx.x % warpsize;
    const int threadColInWarp = threadIdxInWarp % (WSUBN / TN);
    const int threadRowInWarp = threadIdxInWarp / (WSUBN / TN);
    //int thread_In_Warp_stride = WSUBN / TN; 


    //int tx = threadIdx.x % (threads_row_block) * TN;
    //int ty = threadIdx.x / (threads_row_block) * TM;

    int ldg_a_num = BM * BK / (threads_num * 4);
    int ldg_b_num = BK * BN / (threads_num * 4);

    int a_tile_row = threadIdx.x / (BK / 4); // one thread read one float4 once, which has four float
    int a_tile_col = threadIdx.x % (BK / 4) * 4;  // the col index increases from 0 to 4, 8 ....
    int a_tile_stride = BM / ldg_a_num;

    int b_tile_row = threadIdx.x / (BN / 4);
    int b_tile_col = threadIdx.x % (BN / 4) * 4;
    int b_tile_stride = BK / ldg_b_num;


    A += by * BM * n;
    B += bx * BN;
    C += (by * BM + warprow * WM )* n + bx * BN + warpcol * WN;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BM];

    float A_reg[WMITER * TM];
    float B_reg[WNITER * TN];
    float tmp[WMITER * WNITER * TM * TN] = {0.0};

    for (auto k0 = 0; k0 < k; k0 += BK)
    {
        // fill As and Bs
        for (auto i = 0; i < BM; i+=a_tile_stride)
        {
            float4 tmpAs = reinterpret_cast<float4*>(&(A[(a_tile_row + i) * k + a_tile_col]))[0];
            As[a_tile_col * BM + a_tile_row + i]  = tmpAs.x;
            As[(a_tile_col + 1) * BM + a_tile_row + i]  = tmpAs.y;
            As[(a_tile_col + 2) * BM + a_tile_row + i]  = tmpAs.z;
            As[(a_tile_col + 3) * BM + a_tile_row + i]  = tmpAs.w;
            //if (bx == 0 && by == 0 && threadRowInWarp == 0 && threadColInWarp == 0 && warprow == 0 && warpcol == 0)
            //printf("tmp = %f, %f %f %f\n", tmpAs.x, tmpAs.y , tmpAs.z, tmpAs.w);
        }

        for (auto j = 0; j < BK; j += b_tile_stride)
        {
            reinterpret_cast<float4*>(&(Bs[(b_tile_row + j) * BN + b_tile_col]))[0] = reinterpret_cast<float4*>(&(B[(b_tile_row + j) * n + b_tile_col]))[0];
        }

        __syncthreads();
        

        // load needed data from shared memory
        for (auto p = 0; p < BK; p++)
        {
            for (auto wSubRowIdx = 0; wSubRowIdx < WMITER; wSubRowIdx++)
            {
                for (auto i0 = 0; i0 < TM; i0+=4)
                {
                    reinterpret_cast<float4*>(&(A_reg[wSubRowIdx * TM + i0]))[0] = reinterpret_cast<float4*>(&(As[p * BM + warprow *  WM + wSubRowIdx * WSUBM + threadRowInWarp * TM + i0]))[0];  
                }
            }

            for (auto wSubColIdx = 0; wSubColIdx < WNITER; wSubColIdx++)
            {
                for (auto j0 = 0; j0 < TN; j0+=4)
                {
                    reinterpret_cast<float4*>(&(B_reg[wSubColIdx * TN + j0]))[0] = reinterpret_cast<float4*>(&(Bs[p * BN + warpcol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + j0]))[0];
                }
            }


            for (auto wSubRowIdx = 0; wSubRowIdx < WMITER; wSubRowIdx++)
            {
                for (auto i0 = 0; i0 < TM; i0++)
                {
                    for (auto wSubColIdx = 0; wSubColIdx < WNITER; wSubColIdx++)
                    {
                        for (auto j0 = 0; j0 < TN; j0++)
                        {
                            tmp[(wSubRowIdx * TM + i0) * (WNITER * TN) + (wSubColIdx * TN) + j0] += A_reg[wSubRowIdx * TM + i0] * B_reg[wSubColIdx * TN + j0];
                        }
                    }
                }
            }
        }

        A += BK;
        B += BK * n;
        __syncthreads();
        // write the reult back to C
    }

    for (auto wSubRowIdx = 0; wSubRowIdx < WMITER; wSubRowIdx++)
    {
        for (auto i0 = 0; i0 < TM; i0++)
        {
            for (auto wSubColIdx = 0; wSubColIdx < WNITER; wSubColIdx++)
            {
                for (auto j0 = 0; j0 < TN; j0+=4)
                {
                    //float4 intertmp = reinterpret_cast<float4*>(&(C[(by * BM + warprow * WM +  wSubRowIdx * WSUBM + threadRowInWarp * TM + i0) * n + bx * BN + warpcol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + j0]))[0];
                    float4 intertmp = reinterpret_cast<float4*>(&(C[(wSubRowIdx * WSUBM + threadRowInWarp * TM + i0) * n +  wSubColIdx * WSUBN + threadColInWarp * TN + j0]))[0];
                    intertmp.x = alpha * tmp[(wSubRowIdx * TM + i0) * (WNITER * TN) + (wSubColIdx * TN) + j0] + beta * intertmp.x;
                    intertmp.y = alpha * tmp[(wSubRowIdx * TM + i0) * (WNITER * TN) + (wSubColIdx * TN) + j0 + 1] + beta * intertmp.y;
                    intertmp.z = alpha * tmp[(wSubRowIdx * TM + i0) * (WNITER * TN) + (wSubColIdx * TN) + j0 + 2] + beta * intertmp.z;
                    intertmp.w = alpha * tmp[(wSubRowIdx * TM + i0) * (WNITER * TN) + (wSubColIdx * TN) + j0 + 3] + beta * intertmp.w;

                    //reinterpret_cast<float4*>(&(C[(by * BM + warprow * WM +  wSubRowIdx * WSUBM + threadRowInWarp * TM + i0) * n + bx * BN + warpcol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + j0]))[0] = intertmp;
                    reinterpret_cast<float4*>(&(C[(wSubRowIdx * WSUBM + threadRowInWarp * TM + i0) * n +  wSubColIdx * WSUBN + threadColInWarp * TN + j0]))[0] = intertmp;
                }
            }
        }
    }
}
