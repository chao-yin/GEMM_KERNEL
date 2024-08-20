
template<const int BM,
         const int BN,
         const int BK,
         const int TM,
         const int TN>
__global__ void mySgemm_v6(int m, int n, int k, float alpha, float *A, float *B, float beta, float *C)
{
    // calculate thereads num
    const int threads_num = BM * BN / (TM * TN);
    assert(blockDim.x == threads_num);

    // blockId for row and col
    const int brow = blockIdx.y;
    const int bcol = blockIdx.x;

    int ty = threadIdx.x / (BN / TN) * TM;
    int tx = threadIdx.x % (BN / TN) * TN;

    // calculate a_tile parameter
    int a_tile_row = threadIdx.x / (BK / 4);
    int a_tile_col = threadIdx.x % (BK / 4) * 4; // this two parameter is for the data copy
    int a_tile_stride = (threads_num * 4) / BK;

    // calculate b_tile parameter
    int b_tile_row = threadIdx.x / (BN / 4);
    int b_tile_col = threadIdx.x % (BN / 4) * 4;
    int b_tile_stride = (threads_num * 4) / BN;

    A += brow * BM * k;
    B += bcol * BN;
    C += brow * BM * n + bcol * BN;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    float A_reg[TM];
    float B_reg[TN];
    float tmp[TM * TN] = {0.0};

    for (auto k0 = 0; k0 < k; k0+=BK)
    {
        // copy data from global memory to shared_memory
        for (auto i0 = 0; i0 < BM; i0+=a_tile_stride)
        {
            float4 tmpAs = reinterpret_cast<float4*>(&(A[(i0 + a_tile_row) * k + a_tile_col]))[0];
            As[a_tile_col * BM + i0 + a_tile_row] = tmpAs.x;
            As[(a_tile_col + 1) * BM + i0 + a_tile_row] = tmpAs.y;
            As[(a_tile_col + 2) * BM + i0 + a_tile_row] = tmpAs.z;
            As[(a_tile_col + 3) * BM + i0 + a_tile_row] = tmpAs.w;
        }

        // copy B
        for (auto j0 = 0; j0 < BK; j0+=b_tile_stride)
        {
            reinterpret_cast<float4*>(&(Bs[(j0 + b_tile_row) * BN + b_tile_col]))[0] = reinterpret_cast<float4*>(&(B[(j0 + b_tile_row) * n + b_tile_col]))[0];
        }

        __syncthreads();

        A += BK;
        B += BK * n;

        for (auto p = 0; p < BK; p++)
        {
            // load A
            for (auto i0 = 0; i0 < TM; i0+=4)
            {
                reinterpret_cast<float4*>(&(A_reg[i0]))[0] = reinterpret_cast<float4*>(&(As[p * BM + ty + i0]))[0];
            }

            // load B
            for (auto j0 = 0; j0 < TN; j0+=4)
            {
                reinterpret_cast<float4*>(&(B_reg[j0]))[0] = reinterpret_cast<float4*>(&(Bs[p * BN + tx + j0]))[0];
            }

            #pragma unroll
            for (auto i = 0; i < TM; i++)
            {
                for (auto j =0; j < TN; j++)
                {
                    tmp[i * TN + j] += A_reg[i] * B_reg[j];
                }
            }

        }

        __syncthreads();
    }
    for (auto i = 0; i < TM; i++)
    for (auto j = 0; j < TN; j+=4)
    {
        float4 ctmp = reinterpret_cast<float4*>(&(C[(ty + i) * n + tx +j]))[0];
        ctmp.x = alpha * tmp[i * TN + j] + beta * ctmp.x;
        ctmp.y = alpha * tmp[i * TN + j + 1] + beta * ctmp.y;
        ctmp.z = alpha * tmp[i * TN + j + 2] + beta * ctmp.z;
        ctmp.w = alpha * tmp[i * TN + j + 3] + beta * ctmp.w;

        reinterpret_cast<float4*>(&(C[(ty + i) * n + tx + j]))[0] = ctmp;

    }
}

#if 0
template<const int BM,
         const int BN,
         const int BK,
         const int TM,
         const int TN>
__global__ void mySgemm_v6(int m, int n, int k, float alpha, float *A, float *B, float beta, float *C)
{
    int threads_num = BN * BN / (TM * TN);

    assert( blockDim.x == threads_num );


    int brow = blockIdx.y;
    int bcol = blockIdx.x;

    int tx = threadIdx.x % (BN / TN) * TN;
    int ty = threadIdx.x / (BN / TN) * TM;

    int a_tile_row = threadIdx.x / (BK / 4);
    int a_tile_col = threadIdx.x % (BK / 4) * 4;
    int a_tile_stride = 4 * threads_num / BK;

    int b_tile_row = threadIdx.x / (BN / 4);
    int b_tile_col = threadIdx.x % (BN / 4) * 4;
    int b_tile_stride = 4 * threads_num / BN;

    A += brow * BM * k;
    B += bcol * BN;
    C += brow * BM * n + bcol * BN;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    float tmp[TM * TN] = {0.0};
    float A_reg[TM];
    float B_reg[TN];

    for (auto k0 = 0; k0 < k; k0+=BK)
    {
        // copy data A from DRAM to shared memory
        //
        for (auto i0 = 0; i0 < BM; i0+=a_tile_stride)
        {
            float4 tmpAs = reinterpret_cast<float4*>(&(A[(a_tile_row + i0) * k + a_tile_col]))[0];
            As[a_tile_col * BM + a_tile_row + i0] = tmpAs.x;
            As[(a_tile_col + 1) * BM + a_tile_row + i0] = tmpAs.y;
            As[(a_tile_col + 2) * BM + a_tile_row + i0] = tmpAs.z;
            As[(a_tile_col + 3) * BM + a_tile_row + i0] = tmpAs.w;
        }

        // copy data B from DRAM to sahred memory
        //
        for (auto j0 = 0; j0 < BK; j0+=b_tile_stride)
        {
            reinterpret_cast<float4*>(&(Bs[(b_tile_row + j0) * BN + b_tile_col]))[0] = reinterpret_cast<float4*>(&(B[(b_tile_row + j0) * n + b_tile_col]))[0];
        }

        __syncthreads();

        for (auto p = 0; p < BK; p++)
        {
            for (auto i = 0; i < TM; i++)
            {
                A_reg[i] = As[p * BM + ty + i]; // As has been transposed, therefore p is row and (ty + i) is col right now with BM as stride
            }

            for (auto j =0; j < TN; j++)
            {
                B_reg[j] = Bs[p * BN + tx + j];
            }

            for (auto i = 0; i < TM; i++)
            {
                for (auto j = 0; j < TN; j++)
                {
                    tmp[i * TN + j] += A_reg[i] * B_reg[j];
                }
            }

        }
        A += BK;
        B += BK * n;

        __syncthreads();
    }

    for (auto i = 0; i < TM; i++)
    {
        for (auto j = 0; j < TN; j+=4)
        {
            float4 ctmp = reinterpret_cast<float4*>(&(C[(ty + i) * n + tx + j]))[0];
            ctmp.x = alpha * tmp[i * TN + j] + beta * ctmp.x;
            ctmp.y = alpha * tmp[i * TN + j + 1] + beta * ctmp.y;
            ctmp.z = alpha * tmp[i * TN + j + 2] + beta * ctmp.z;
            ctmp.w = alpha * tmp[i * TN + j + 3] + beta * ctmp.w;

            reinterpret_cast<float4*>(&(C[(ty + i) * n + tx + j]))[0] = ctmp;
        }
    }
}
#endif
