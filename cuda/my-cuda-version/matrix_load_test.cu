#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>

#define VECTOR_SIZE 4
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define SIZE 128

template <const int BM = 64, const int BK = 64>
__global__ void copy_matrix(const float *A, const uint M, const uint K, float *B)
{
    // 把A复制到B
    __shared__ float As[BM * BK];
    // 尝试在这个block中将A的数据加载到共享内存中
    // 每个线程加载一点
    // 用float4来加载
    assert(gridDim.x == CEIL_DIV(K, BK));
    assert(gridDim.y == CEIL_DIV(M, BM));
    // 每一个block处理一个BM*BK的矩阵
    // 考虑线程数为32的情况
    // 计算当前block所负责的小的分块矩阵的位置
    const uint row_start = blockIdx.y * BM;
    const uint row_end = (blockIdx.y + 1) * BM;
    const uint col_start = blockIdx.x * BK;
    const uint col_end = (blockIdx.x + 1) * BK;

    // 计算当前线程负责的元素
    const uint thread_row = threadIdx.x / (BK / VECTOR_SIZE);
    const uint thread_col = threadIdx.x % (BK / VECTOR_SIZE);
    const uint repeat_nums = BM * BK / VECTOR_SIZE / blockDim.x; // 线程数是32,每个线程处理4个元素，要处理的数据为64*64个，因此每一个线程要跑64*64/4/32次
    const uint stride = BM / repeat_nums;                        // 总行数处以要重复的次数就是每次处理行间隔的行数
    // 如果当前block的行列都没有超过M和K
    if (row_end <= M && col_end <= K)
    {
        // 全部按照向量化的方式加载，且不会越界
        for (uint row_offset = 0; row_offset < BM; row_offset += stride)
        {
            const uint _As_index = (thread_row + row_offset) * BK + thread_col * VECTOR_SIZE;
            const uint _A_index = (row_start + thread_row + row_offset) * K + col_start + thread_col * VECTOR_SIZE;
            reinterpret_cast<float4 *>(&As[_As_index])[0] = reinterpret_cast<const float4 *>(&A[_A_index])[0];
        }
        // 预期是从A中的一小块数据都一对一加载到了As中
        __syncthreads();
        // 写回去
        for (uint row_offset = 0; row_offset < BM; row_offset += stride)
        {
            const uint _As_index = (thread_row + row_offset) * BK + thread_col * VECTOR_SIZE;
            const uint _B_index = (row_start + thread_row + row_offset) * K + col_start + thread_col * VECTOR_SIZE;
            reinterpret_cast<float4 *>(&B[_B_index])[0] = reinterpret_cast<float4 *>(&As[_As_index])[0];
        }
    }
    else
    {
        printf("越界了\n");
    }
}

void test()
{
    float hA[SIZE * SIZE], hB[SIZE * SIZE];
    for (size_t i = 0; i < SIZE; i++)
    {
        for (size_t j = 0; j < SIZE; j++)
        {
            hA[i * SIZE + j] = i * SIZE + j;
        }
    }

    float *dA, *dB;
    cudaMalloc(&dA, SIZE * SIZE * sizeof(float));
    cudaMalloc(&dB, SIZE * SIZE * sizeof(float));
    cudaMemcpy(dA, hA, SIZE * SIZE * sizeof(float), cudaMemcpyHostToDevice);
    const dim3 gridDim = {CEIL_DIV(SIZE, 64), CEIL_DIV(SIZE, 64)}, blockDim = 32;
    copy_matrix<<<gridDim, blockDim>>>(dA, SIZE, SIZE, dB);
    // 检查cuda有没有出错
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return;
    }
    cudaMemcpy(hB, dB, SIZE * SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    // 等待同步完成
    cudaDeviceSynchronize();
    cudaFree(dA);
    cudaFree(dB);
    for (size_t i = 0; i < SIZE; i++)
    {
        for (size_t j = 0; j < SIZE; j++)
        {
            if (hA[i * SIZE + j] != hB[i * SIZE + j])
            {
                printf("error at %zu %zu: expected %f, got %f\n", i, j, hA[i * SIZE + j], hB[i * SIZE + j]);
                return;
            }
        }
    }
    printf("success\n");
}

int main()
{
    test();
    return 0;
}
