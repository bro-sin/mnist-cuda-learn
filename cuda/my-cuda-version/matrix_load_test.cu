#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>

#define VECTOR_SIZE 4
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define ROW_SIZE 128
#define COL_SIZE 13

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
            const uint _A_index = (row_start + thread_row + row_offset) * K + col_start + thread_col * VECTOR_SIZE; // 有可能这个index不是按16字节对齐的？
            const float *_tmp_A = &A[_A_index];
            const uintptr_t _tmp_A_ptr = reinterpret_cast<const uintptr_t>(_tmp_A);
            const uint _float4_size = sizeof(float4);
            // if (_tmp_A_ptr % _float4_size == 0)
            if (_A_index % VECTOR_SIZE == 0)
            {
                if (_As_index % VECTOR_SIZE != 0)
                {
                    printf("共享没有对齐");
                }

                reinterpret_cast<float4 *>(&As[_As_index])[0] = reinterpret_cast<const float4 *>(&A[_A_index])[0];
            }
            else
            {
                // 如果不是16字节对齐的，那么就要一个一个元素的加载
                for (uint _vec_index = 0; _vec_index < VECTOR_SIZE; _vec_index++)
                {
                    As[_As_index + _vec_index] = A[_A_index + _vec_index];
                }
                // 输出是哪个线程，哪一行，哪一列出现的这个情况
                // printf("threadIdx.x: %d, thread_row: %d, thread_col: %d, A_row:%d, A_col:%d\n", threadIdx.x, thread_row, thread_col, row_start + thread_row + row_offset, col_start + thread_col * VECTOR_SIZE);
            }
        }
        // 预期是从A中的一小块数据都一对一加载到了As中
        __syncthreads();
        // 写回去
        for (uint row_offset = 0; row_offset < BM; row_offset += stride)
        {
            const uint _As_index = (thread_row + row_offset) * BK + thread_col * VECTOR_SIZE;
            const uint _B_index = (row_start + thread_row + row_offset) * K + col_start + thread_col * VECTOR_SIZE;
            // reinterpret_cast<float4 *>(&B[_B_index])[0] = reinterpret_cast<float4 *>(&As[_As_index])[0];
            const float *_tmp_As = &As[_As_index];
            const uintptr_t _tmp_As_ptr = reinterpret_cast<const uintptr_t>(_tmp_As);
            if (_B_index % VECTOR_SIZE == 0)
            {
                reinterpret_cast<float4 *>(&B[_B_index])[0] = reinterpret_cast<float4 *>(&As[_As_index])[0];
            }
            else
            {
                for (uint _vec_index = 0; _vec_index < VECTOR_SIZE; _vec_index++)
                {
                    B[_B_index + _vec_index] = As[_As_index + _vec_index];
                }
            }
        }
        __syncthreads();
    }
    else if (row_end <= M && col_end > K)
    {
        // 行没有越界，列有部分越界了
        for (uint row_offset = 0; row_offset < BM; row_offset += stride)
        {
            const uint _As_index = (thread_row + row_offset) * BK + thread_col * VECTOR_SIZE;
            const uint _A_index = (row_start + thread_row + row_offset) * K + col_start + thread_col * VECTOR_SIZE;
            for (uint _vec_index = 0; _vec_index < VECTOR_SIZE; _vec_index++)
            {
                if (col_start + thread_col * VECTOR_SIZE + _vec_index < K) // 如果列没有越界
                {
                    As[_As_index + _vec_index] = A[_A_index + _vec_index];
                }
                else
                {
                    As[_As_index + _vec_index] = 0;
                }
            }
        }
        __syncthreads();
        for (uint row_offset = 0; row_offset < BM; row_offset += stride)
        {
            const uint _As_index = (thread_row + row_offset) * BK + thread_col * VECTOR_SIZE;
            const uint _B_index = (row_start + thread_row + row_offset) * K + col_start + thread_col * VECTOR_SIZE;
            for (uint _vec_index = 0; _vec_index < VECTOR_SIZE; _vec_index++)
            {
                if (col_start + thread_col * VECTOR_SIZE + _vec_index < K)
                {
                    B[_B_index + _vec_index] = As[_As_index + _vec_index];
                }
            }
        }
        __syncthreads();
    }
    else
    {
        printf("还没做呢\n");
    }
}

void test()
{
    float hA[ROW_SIZE * COL_SIZE], hB[ROW_SIZE * COL_SIZE];
    for (size_t i = 0; i < ROW_SIZE; i++)
    {
        for (size_t j = 0; j < COL_SIZE; j++)
        {
            hA[i * COL_SIZE + j] = i * COL_SIZE + j;
        }
    }

    float *dA, *dB;
    cudaMalloc(&dA, ROW_SIZE * COL_SIZE * sizeof(float));
    cudaMalloc(&dB, ROW_SIZE * COL_SIZE * sizeof(float));
    cudaMemcpy(dA, hA, ROW_SIZE * COL_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    const dim3 gridDim = {CEIL_DIV(COL_SIZE, 64), CEIL_DIV(ROW_SIZE, 64)}, blockDim = 32;
    copy_matrix<<<gridDim, blockDim>>>(dA, ROW_SIZE, COL_SIZE, dB);
    // 检查cuda有没有出错
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return;
    }
    cudaMemcpy(hB, dB, ROW_SIZE * COL_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    // 等待同步完成
    cudaDeviceSynchronize();
    cudaFree(dA);
    cudaFree(dB);
    for (size_t i = 0; i < ROW_SIZE; i++)
    {
        for (size_t j = 0; j < COL_SIZE; j++)
        {
            if (hA[i * COL_SIZE + j] != hB[i * COL_SIZE + j])
            {
                printf("error at %zu %zu: expected %f, got %f\n", i, j, hA[i * COL_SIZE + j], hB[i * COL_SIZE + j]);
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
