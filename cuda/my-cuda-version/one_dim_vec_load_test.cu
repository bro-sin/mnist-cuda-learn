#include <cuda_runtime.h>
#include <stdio.h>

#define VECTOR_SIZE 4
#define SIZE 81
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

__global__ void copy_vec(const float *A, const uint length, float *B)
{
    // copy data from A to B
    // only one block
    // using shared memory
    __shared__ float As[16 * 16];
    // 将长度拆分为16*16的小的向量
    // 考虑一个block为32个线程，一次处理32*4个向量
    const uint idx = threadIdx.x;
    const uint stride = 16 * 16 / VECTOR_SIZE / blockDim.x; // 要重复的次数，应该是2,每个线程处理两个元素
    uint len_index;
    for (len_index = 0; len_index < length; len_index += 16 * 16)
    {
        for (uint offset = 0; offset < stride; offset++)
        {
            reinterpret_cast<float4 *>(&As[idx * VECTOR_SIZE + offset * blockDim.x * VECTOR_SIZE])[0] =
                reinterpret_cast<const float4 *>(&A[len_index + idx * VECTOR_SIZE + offset * blockDim.x * VECTOR_SIZE])[0];
        }
        __syncthreads();
        // 写回去
        for (uint offset = 0; offset < stride; offset++)
        {
            reinterpret_cast<float4 *>(&B[len_index + idx * VECTOR_SIZE + offset * blockDim.x * VECTOR_SIZE])[0] =
                reinterpret_cast<const float4 *>(&As[idx * VECTOR_SIZE + offset * blockDim.x * VECTOR_SIZE])[0];
        }
        __syncthreads();
    }
    // 还有剩下的
    for (len_index = length - length % (16 * 16); len_index < length; len_index++)
    {
        B[len_index] = A[len_index];
    }
}

void test_copy_vec()
{
    float hA[SIZE * SIZE], hB[SIZE * SIZE];
    for (size_t i = 0; i < SIZE; i++)
    {
        for (size_t j = 0; j < SIZE; j++)
        {
            hA[i * SIZE + j] = static_cast<float>(i * SIZE + j);
        }
    }
    float *dA, *dB;
    cudaMalloc(&dA, SIZE * SIZE * sizeof(float));
    cudaMalloc(&dB, SIZE * SIZE * sizeof(float));
    cudaMemcpy(dA, hA, SIZE * SIZE * sizeof(float), cudaMemcpyHostToDevice);
    copy_vec<<<1, 32>>>(dA, SIZE * SIZE, dB);
    cudaMemcpy(hB, dB, SIZE * SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < SIZE; i++)
    {
        for (size_t j = 0; j < SIZE; j++)
        {
            if (hA[i * SIZE + j] != hB[i * SIZE + j])
            {
                printf("Error: %f %f\n", hA[i * SIZE + j], hB[i * SIZE + j]);
                return;
            }
        }
    }
    printf("Success\n");
}

int main()
{
    test_copy_vec();
    return 0;
}