#include "cuUtils.hpp"

__global__ void _unravel(uchar *src, uchar *dst1, uchar *dst2, int N)
{
    // int i = (blockDim.x * threadIdx.y) + threadIdx.x;    

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N)
    {
        if (i % 2 == 0)
        {
            int idx = i / 2;
            dst1[idx] = src[i];
        }
        else
        {
            int idx = (i - 1) / 2;
            dst2[idx] = src[i];
        }
    }
}


// void unravel(uchar *src,
//              uchar *dst1,
//              uchar *dst2,
//              int width,
//              int height)
// {
//     dim3 threadsPerBlock(width, height);

//     _unravel<<<1, threadsPerBlock>>>(src, dst1, dst2, width * height);
// }


void unravel(uchar *src,
             uchar *dst1,
             uchar *dst2,
             int width,
             int height)
{
    // int MAX_BLOCK_SIZE = 32;
    // dim3 threadsPerBlock(MAX_BLOCK_SIZE, MAX_BLOCK_SIZE);

    // int N = width * height;
    // int threads = MAX_BLOCK_SIZE * MAX_BLOCK_SIZE;
    // int blocksPerGrid = (N + threads - 1) / threads;

    // _unravel<<<blocksPerGrid, threadsPerBlock>>>(src, dst1, dst2, width * height);

    int N = width * height;
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    _unravel<<<blocksPerGrid, threadsPerBlock>>>(src, dst1, dst2, N);

}