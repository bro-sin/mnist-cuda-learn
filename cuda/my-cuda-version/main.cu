#include <cstdio>
#include <cassert>
#include <iostream>
#include <string>
#include <concepts>
#include <cuda_runtime.h>

namespace CONSTANTS
{
    const uint PICTURE_WIDTH = 28;
    const uint PICTURE_HEIGHT = 28;
    const uint PICTURE_SIZE = PICTURE_WIDTH * PICTURE_HEIGHT;
    const uint LABEL_SIZE = 1; // 一张图片的label就是一个数
    const uint NUM_CLASSES = 10;
    const uint TRAIN_SIZE = static_cast<uint>(6e4);
    const uint TEST_SIZE = static_cast<uint>(1e4);
}

namespace HYPERPARAMETERS
{
    const uint HIDDEN_SIZE = 256;
    const uint BATCH_SIZE = 4; // 每一次训练使用的样本数
    const float LEARNING_RATE = 1e-3;
    const uint EPOCHS = 4;
}

namespace CUDA_KERNELS
{
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
    // 矩阵乘法，默认A,B,C是行优先存储，
    // C=alpha*A*B+beta*C
    template <const uint BM, const uint BN, const uint BK, const uint TM, const uint TN>
    __global__ void matmul_row_major(
        const uint M,
        const uint N,
        const uint K,
        const float alpha,
        const float *A,
        const float *B,
        const float beta,
        float *C)
    {
        const dim3 small_C_dim = {CEIL_DIV(BN, TN), CEIL_DIV(BM, TM)};
        /*
        small_C_dim是这个block要计算的C(M*N)的一个小块(BM*BN)按照TM*TN分块之后的维度
        */

        const uint threadRow = threadIdx.x / small_C_dim.x;
        const uint threadColumn = threadIdx.x % small_C_dim.x;
        /*
        这个线程计算的小矩阵的行数和列数范围是
        行：[threadRow * TM, threadRow * TM + TM)
        列：[threadColumn * TN, threadColumn * TN + TN)
        */
        float threadResults[TM * TN] = {0.0}; // 这个线程计算的小矩阵的所有元素

        const uint currentRow = blockIdx.y;
        const uint currentColumn = blockIdx.x;
        /*
        currentRow和currentColumn是这个block要计算的C(M*N)的一个小块(BM*BN)的这个矩阵的索引
        */

        A += currentRow * BM * K;
        const uint global_A_row = currentRow * BM;
        uint global_A_col = 0;
        B += currentColumn * BN;
        uint global_B_row = 0;
        const uint global_B_col = currentColumn * BN;

        C += currentRow * BM * N + currentColumn * BN;
        const uint global_C_row = currentRow * BM;
        const uint global_C_col = currentColumn * BN;

        __shared__ float As[BM * BK];
        __shared__ float Bs[BK * BN];
        /*
        一个block的线程一共有small_C_dim.x * small_C_dim.y个,
        这个值会比BM*BK以及BK*BN小，因此没办法每个线程加载一个元素就将所有元素加载到共享内存中
        可以隔几行加载一个元素，
        但不能隔固定位置加载一个元素，固定的偏差在大矩阵中向下移动的距离是不一样的
        这里隔的行数就是strideA和strideB
        这个也是一个block的线程一次可以加载的行数
        一个block一次加载stride行，那么有row行，就加载row/stride次
        */

        const uint innerRowA = threadIdx.x / BK;
        const uint innerColumnA = threadIdx.x % BK;
        const uint strideA = small_C_dim.x * small_C_dim.y / BK;

        const uint innerRowB = threadIdx.x / BN;
        const uint innerColumnB = threadIdx.x % BN;
        const uint strideB = small_C_dim.x * small_C_dim.y / BN;

        const uint outer_dot_nums = CEIL_DIV(K, BK);

        // float register_cache_A;
        float register_cache_A[TM] = {0};
        float register_cache_B[TN] = {0};

        for (uint outer_dot_index = 0; outer_dot_index < outer_dot_nums; outer_dot_index++)
        {
            for (uint _row_load_offset = 0; _row_load_offset < BM; _row_load_offset += strideA)
            {
                const uint _global_A_row_offset = innerRowA + _row_load_offset;
                const uint _global_A_col_offset = innerColumnA;
                if (global_A_col + _global_A_col_offset < K && global_A_row + _global_A_row_offset < M)
                {
                    // 加载A并转置后存到As
                    As[innerColumnA * BM + _global_A_row_offset] =
                        A[(_global_A_row_offset)*K + innerColumnA];
                }
                else
                {
                    As[innerColumnA * BM + _global_A_row_offset] = 0.0;
                }
            }

            for (uint _row_load_offset = 0; _row_load_offset < BK; _row_load_offset += strideB)
            {
                const uint _global_B_row_offset = innerRowB + _row_load_offset;
                const uint _global_B_col_offset = innerColumnB;
                if (global_B_col + _global_B_col_offset < N && global_B_row + _global_B_row_offset < K)
                {
                    Bs[(_global_B_row_offset)*BN + innerColumnB] =
                        B[(_global_B_row_offset)*N + innerColumnB];
                }
                else
                {
                    Bs[(_global_B_row_offset)*BN + innerColumnB] = 0.0;
                }
            }

            __syncthreads();

            A += BK;
            global_A_col += BK;
            B += BK * N;
            global_B_row += BK;

            for (uint inner_dot_index = 0; inner_dot_index < BK; inner_dot_index++)
            {
                for (uint resIdxN = 0; resIdxN < TN; resIdxN++)
                {
                    // 取Bs的第inner_dot_index*BN行的threadColumn*TN+resIdxN列
                    register_cache_B[resIdxN] = Bs[inner_dot_index * BN + threadColumn * TN + resIdxN];
                }
                for (uint resIdxM = 0; resIdxM < TM; resIdxM++)
                {
                    // 取As的数据，但是要转置回去
                    register_cache_A[resIdxM] = As[inner_dot_index * BM + threadRow * TM + resIdxM];

                    for (uint resIdxN = 0; resIdxN < TN; resIdxN++)
                    {
                        threadResults[resIdxM * TN + resIdxN] +=
                            register_cache_A[resIdxM] *
                            register_cache_B[resIdxN];
                    }
                }
            }
            __syncthreads();
        }

        for (uint resIdxM = 0; resIdxM < TM; resIdxM++)
        {
            for (uint resIdxN = 0; resIdxN < TN; resIdxN++)
            {
                const uint _global_C_row_offset = threadRow * TM + resIdxM;
                const uint _global_C_col_offset = threadColumn * TN + resIdxN;
                if (global_C_row + _global_C_row_offset < M && global_C_col + _global_C_col_offset < N)
                {
                    C[_global_C_row_offset * N + _global_C_col_offset] =
                        alpha * threadResults[resIdxM * TN + resIdxN] +
                        beta * C[_global_C_row_offset * N + _global_C_col_offset];
                }
            }
        }
    }

    // 矩阵乘法，A是列优先排列，B,C是行优先排列
    // C=alpha*A*B+beta*C
    template <const uint BM, const uint BN, const uint BK, const uint TM, const uint TN>
    __global__ void matmul_with_A_col_major_and_B_C_row_major(
        const uint M,
        const uint N,
        const uint K,
        const float alpha,
        const float *A, // 列优先
        const float *B,
        const float beta,
        float *C)
    {
        const dim3 small_C_dim = {CEIL_DIV(BN, TN), CEIL_DIV(BM, TM)};
        /*
        small_C_dim是这个block要计算的C(M*N)的一个小块(BM*BN)按照TM*TN分块之后的维度
        */

        const uint threadRow = threadIdx.x / small_C_dim.x;
        const uint threadColumn = threadIdx.x % small_C_dim.x;
        /*
        这个线程计算的小矩阵的行数和列数范围是
        行：[threadRow * TM, threadRow * TM + TM)
        列：[threadColumn * TN, threadColumn * TN + TN)
        */
        float threadResults[TM * TN] = {0.0}; // 这个线程计算的小矩阵的所有元素

        const uint currentRow = blockIdx.y;
        const uint currentColumn = blockIdx.x;
        /*
        currentRow和currentColumn是这个block要计算的C(M*N)的一个小块(BM*BN)的这个矩阵的索引
        */

        A += currentRow * BM;
        const uint global_A_row = currentRow * BM;
        uint global_A_col = 0;
        B += currentColumn * BN;
        uint global_B_row = 0;
        const uint global_B_col = currentColumn * BN;

        C += currentRow * BM * N + currentColumn * BN;
        const uint global_C_row = currentRow * BM;
        const uint global_C_col = currentColumn * BN;

        // 这里都使用行优先排列
        __shared__ float As[BM * BK];
        __shared__ float Bs[BK * BN];
        /*
        一个block的线程一共有small_C_dim.x * small_C_dim.y个,
        这个值会比BM*BK以及BK*BN小，因此没办法每个线程加载一个元素就将所有元素加载到共享内存中
        可以隔几行加载一个元素，
        但不能隔固定位置加载一个元素，固定的偏差在大矩阵中向下移动的距离是不一样的
        这里隔的行数就是strideA和strideB
        这个也是一个block的线程一次可以加载的行数
        一个block一次加载stride行，那么有row行，就加载row/stride次
        */

        const uint innerRowA = threadIdx.x / BK;
        const uint innerColumnA = threadIdx.x % BK;
        const uint strideA = small_C_dim.x * small_C_dim.y / BK;

        const uint innerRowB = threadIdx.x / BN;
        const uint innerColumnB = threadIdx.x % BN;
        const uint strideB = small_C_dim.x * small_C_dim.y / BN;

        const uint outer_dot_nums = CEIL_DIV(K, BK);

        // float register_cache_A;
        float register_cache_A[TM] = {0};
        float register_cache_B[TN] = {0};

        for (uint outer_dot_index = 0; outer_dot_index < outer_dot_nums; outer_dot_index++)
        {
            for (uint _row_load_offset = 0; _row_load_offset < BM; _row_load_offset += strideA)
            {
                const uint _global_A_row_offset = innerRowA + _row_load_offset;
                const uint _global_A_col_offset = innerColumnA;
                if (global_A_col + _global_A_col_offset < K && global_A_row + _global_A_row_offset < M)
                {
                    // 加载A并转置后存到As
                    As[innerColumnA * BM + _global_A_row_offset] =
                        A[_global_A_col_offset * M + _global_A_row_offset];
                }
                else
                {
                    As[innerColumnA * BM + _global_A_row_offset] = 0.0;
                }
            }

            for (uint _row_load_offset = 0; _row_load_offset < BK; _row_load_offset += strideB)
            {
                const uint _global_B_row_offset = innerRowB + _row_load_offset;
                const uint _global_B_col_offset = innerColumnB;
                if (global_B_col + _global_B_col_offset < N && global_B_row + _global_B_row_offset < K)
                {
                    Bs[(_global_B_row_offset)*BN + innerColumnB] =
                        B[(_global_B_row_offset)*N + innerColumnB];
                }
                else
                {
                    Bs[(_global_B_row_offset)*BN + innerColumnB] = 0.0;
                }
            }

            __syncthreads();

            A += BK * M;
            global_A_col += BK;
            B += BK * N;
            global_B_row += BK;

            for (uint inner_dot_index = 0; inner_dot_index < BK; inner_dot_index++)
            {
                for (uint resIdxN = 0; resIdxN < TN; resIdxN++)
                {
                    // 取Bs的第inner_dot_index*BN行的threadColumn*TN+resIdxN列
                    register_cache_B[resIdxN] = Bs[inner_dot_index * BN + threadColumn * TN + resIdxN];
                }
                for (uint resIdxM = 0; resIdxM < TM; resIdxM++)
                {
                    // 取As的数据，但是要转置回去
                    register_cache_A[resIdxM] = As[inner_dot_index * BM + threadRow * TM + resIdxM];

                    for (uint resIdxN = 0; resIdxN < TN; resIdxN++)
                    {
                        threadResults[resIdxM * TN + resIdxN] +=
                            register_cache_A[resIdxM] *
                            register_cache_B[resIdxN];
                    }
                }
            }
            __syncthreads();
        }

        for (uint resIdxM = 0; resIdxM < TM; resIdxM++)
        {
            for (uint resIdxN = 0; resIdxN < TN; resIdxN++)
            {
                const uint _global_C_row_offset = threadRow * TM + resIdxM;
                const uint _global_C_col_offset = threadColumn * TN + resIdxN;
                if (global_C_row + _global_C_row_offset < M && global_C_col + _global_C_col_offset < N)
                {
                    C[_global_C_row_offset * N + _global_C_col_offset] =
                        alpha * threadResults[resIdxM * TN + resIdxN] +
                        beta * C[_global_C_row_offset * N + _global_C_col_offset];
                }
            }
        }
    }

    // B=A*scale
    __global__ void matmul_by_scale_same_major_axis(
        const float *A,
        const uint M,
        const uint N,
        const float scale,
        float *B)
    {
        const uint row = blockIdx.y * blockDim.y + threadIdx.y;
        const uint col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < M && col < N)
        {
            B[row * N + col] = A[row * N + col] * scale;
        }
    }

    __global__ void matadd_row_major(const float *A,
                                     const float *B,
                                     const uint M,
                                     const uint N,
                                     float *C)
    {
        const uint row = blockIdx.y * blockDim.y + threadIdx.y;
        const uint col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < M && col < N)
        {
            C[row * N + col] = A[row * N + col] + B[row * N + col];
        }
    }

    // A+=B
    __global__ void matadd_assign_row_major(float *A,
                                            const float *B,
                                            const uint M,
                                            const uint N)
    {
        const uint row = blockIdx.y * blockDim.y + threadIdx.y;
        const uint col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < M && col < N)
        {
            A[row * N + col] += B[row * N + col];
        }
    }

    // A+=bias
    __global__ void matadd_assign_bias_row_major(float *A,
                                                 const float *bias, //[M,1]
                                                 const uint M,
                                                 const uint N // N在这里为batch_size
    )
    {
        const uint row = blockIdx.y * blockDim.y + threadIdx.y;
        const uint col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < M && col < N)
        {
            A[row * N + col] += bias[row];
        }
    }

    // A,B都是行优先
    // 按行的方向求和（消去行的维度）
    __global__ void sum_row_major_by_row(
        const float *A, // A[M*N]
        const uint M,
        const uint N,
        float *B // B[N]
    )
    {
        // 每个线程负责B的一个元素
        const uint col = (blockIdx.y * blockDim.y + threadIdx.y) * (gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);
        if (col < N)
        {
            float tmp_sum_result = 0.0f;
            for (uint row = 0; row < M; row++)
            {
                tmp_sum_result += A[row * N + col];
            }
            B[col] = tmp_sum_result;
        }
    }

    // A,B都是行优先
    // 按列的方向求和（消去列的维度）
    __global__ void sum_row_major_by_col(
        const float *A, // A[M*N]
        const uint M,
        const uint N,
        float *B // B[M]
    )
    {
        // 每个线程负责B的一个元素
        const uint row = (blockIdx.y * blockDim.y + threadIdx.y) * (gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);
        if (row < M)
        {
            float tmp_sum_result = 0.0f;
            for (uint col = 0; col < N; col++)
            {
                tmp_sum_result += A[row * N + col];
            }
            B[row] = tmp_sum_result;
        }
    }

    // A和B的数据排列顺序是一致的，例如都是行优先
    __global__ void mat_relu(const float *A, const uint length, float *B)
    {

        const uint index = (blockIdx.y * blockDim.y + threadIdx.y) * (gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);
        if (index < length)
        {
            B[index] = fmaxf(0.0f, A[index]);
        }
    }

    // input,grad_output,grad_input都是行优先
    __global__ void mat_relu_backward(const float *input, const uint length, const float *grad_output, float *grad_input)
    {

        const uint index = (blockIdx.y * blockDim.y + threadIdx.y) * (gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);
        if (index < length)
        {
            if (input[index] > 0)
            {
                grad_input[index] = grad_output[index];
            }
            else
            {
                grad_input[index] = 0.0f;
            }
        }
    }

    __global__ void softmax_row_major(
        const float *input, const uint M, const uint N, float *output)
    {
        const uint row = blockIdx.y * blockDim.y + threadIdx.y;
        const uint col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < M && col < N)
        {
            float max_val = input[col];
            for (uint i = 1; i < M; i++)
            {
                max_val = fmaxf(max_val, input[i * N + col]);
            }

            float sum = 0.0f;
            for (uint i = 0; i < M; i++)
            {
                sum += expf(input[i * N + col] - max_val);
            }

            // TODO: 将上面找最大值和计算sum的过程分配给一个block中的所有线程去做

            output[row * N + col] = expf(input[row * N + col] - max_val) / sum;
        }
    }

    // N:batch_size, M:class_nums
    __global__ void cross_entropy_loss_row_major(
        const float *input, // 要求是先经过softmax的,[M*N]
        const uint M,
        const uint N,
        const int *target, //[1*N]
        float *loss        // 最终计算完成后*loss就是损失值
    )
    {
        const uint col_index = (blockIdx.y * blockDim.y + threadIdx.y) * (gridDim.x * blockDim.x) + (blockIdx.x * blockDim.x + threadIdx.x);
        if (col_index < N)
        {
            uint max_prob_index = target[col_index];
            float max_prob = input[max_prob_index * N + col_index];
            // 对max_prob取对数，然后取平均然后取相反数
            // 这里交换顺序，先去对数，然后相反数，然后处以N,最后全部加起来
            atomicAdd(loss, -logf(max_prob) / N);
        }
    }
}

namespace Math
{
    enum Axis
    {
        Row,
        Column,
    };

    template <typename T>
    concept Arithmetic = requires(T a, T b) {
        { a + b } -> std::convertible_to<T>;
        { a - b } -> std::convertible_to<T>;
        { a * b } -> std::convertible_to<T>;
        { a / b } -> std::convertible_to<T>;
    };

    template <Arithmetic MatrixDataType = float>
    class Matrix
    {
    public:
        MatrixDataType *data;
        mutable MatrixDataType *gpu_device_data = nullptr;
        uint rows_num;
        uint cols_num;
        const bool is_external_data = false; // 用于判断data是外界传入的还是内部创建的，默认数据是内部创建的，应当在析构函数中释放内存
        const bool is_external_gpu_device_data = false;
        Axis major_axis = Axis::Row;
        mutable bool use_gpu = false;

        Matrix(); // 完全用于内部创建对象

        uint get_index(uint row, uint col) const;

    public:
        Matrix(MatrixDataType *data,
               uint rows_num,
               uint cols_num,
               Axis major_axis = Axis::Row); // 给外面调用的构造函数，保持is_external_data为true
        Matrix(MatrixDataType *data,
               MatrixDataType *gpu_device_data, // 这个构造函数应当保持is_external_gpu_device_data为true
               uint rows_num,
               uint cols_num,
               Axis major_axis = Axis::Row); // 给外面调用的构造函数，保持is_external_data为true
        Matrix(const Matrix<MatrixDataType> &matrix);
        ~Matrix();

        void init_cuda() const;
        void cuda() const; // TODO: 设计使用cuda之后，数据指针采用gpu上的，另外有类似的cpu版本
        void copy_host_to_device() const;
        void copy_device_to_host();

        void show() const;
        std::string get_matrix_info() const;

        MatrixDataType get_item(uint row, uint col) const;

        template <Arithmetic OutputMatrixDataType = MatrixDataType>
        Matrix<OutputMatrixDataType> zeros_like() const; // 当前矩阵是模板,返回矩阵类型默认与模板类型相同

        template <Arithmetic TemplateMatrixDataType>
        static Matrix<MatrixDataType> zeros_like(const Matrix<TemplateMatrixDataType> &matrix); // 不使用类模板参数的时候创建的矩阵是float,使用模板参数就以模板类型为准

        static Matrix<MatrixDataType> zeros(uint rows_num, uint cols_num);
        static Matrix<MatrixDataType> ones(uint rows_num, uint cols_num);

        template <Arithmetic OutputMatrixDataType = MatrixDataType>
        static Matrix<OutputMatrixDataType> ones_like(const Matrix<MatrixDataType> &matrix);

        void transpose();
        Matrix<MatrixDataType> clone() const;
        Matrix<MatrixDataType> get_transpose_matrix() const;
        Matrix<MatrixDataType> get_tmp_transpose_matrix_with_ref_data() const;

        Matrix<MatrixDataType> multiply(const Matrix<MatrixDataType> &matrix) const;
        void multiply(const Matrix<MatrixDataType> &input, Matrix<MatrixDataType> &output) const;
        void multiply(const float scale, Matrix<MatrixDataType> &output) const;

        Matrix<MatrixDataType> add(const Matrix<MatrixDataType> &matrix) const;
        void add(const Matrix<MatrixDataType> &input, Matrix<MatrixDataType> &output) const;

        void add_assign(const Matrix<MatrixDataType> &matrix);

        void sum(Matrix<MatrixDataType> &output, const Axis axis) const;

        void relu(Matrix<MatrixDataType> &output) const;
        void relu();
        Matrix<MatrixDataType> get_relu_matrix() const;
    };
}

namespace Math
{
    // impl Matrix
    template <Arithmetic MatrixDataType>
    Matrix<MatrixDataType>::Matrix(
        // 外部传入数据的构造函数
        MatrixDataType *data,
        uint rows_num,
        uint cols_num,
        Axis major_axis)
        : data(data),
          rows_num(rows_num),
          cols_num(cols_num),
          major_axis(major_axis),
          is_external_data(true)
    {
    }

    template <Arithmetic MatrixDataType>
    Matrix<MatrixDataType>::Matrix(
        // 外部传入数据的构造函数
        MatrixDataType *data,
        MatrixDataType *gpu_device_data,
        uint rows_num,
        uint cols_num,
        Axis major_axis)
        : data(data),
          gpu_device_data(gpu_device_data),
          rows_num(rows_num),
          cols_num(cols_num),
          major_axis(major_axis),
          is_external_data(true),
          is_external_gpu_device_data(true),
          use_gpu(true)
    {
    }

    template <Arithmetic MatrixDataType>
    Matrix<MatrixDataType>::Matrix()
    {
        // 内部构造函数，is_external_data默认为false
        // 其他属性自己设置
    }

    template <Arithmetic MatrixDataType>
    Matrix<MatrixDataType>::Matrix(const Matrix<MatrixDataType> &matrix)
        : rows_num(matrix.rows_num), cols_num(matrix.cols_num), major_axis(matrix.major_axis)
    {
        this->data = new MatrixDataType[this->rows_num * this->cols_num];
        // 将matrix的data复制给this->data
        std::copy(matrix.data, matrix.data + this->rows_num * this->cols_num, this->data);
    }

    template <Arithmetic MatrixDataType>
    Matrix<MatrixDataType>::~Matrix()
    {
        if (!this->is_external_data)
        { // 如果是内部创建的data，需要释放内存
            delete[] this->data;
        }
        this->data = nullptr;
        if (!this->is_external_gpu_device_data)
        {
            cudaFree(this->gpu_device_data);
        }
        this->gpu_device_data = nullptr;
    }

    template <Arithmetic MatrixDataType>
    void Matrix<MatrixDataType>::init_cuda() const
    {
        assert(!this->is_external_gpu_device_data && this->gpu_device_data == nullptr);
        cudaError_t malloc_result = cudaMalloc(&this->gpu_device_data, this->rows_num * this->cols_num * sizeof(MatrixDataType));
        if (malloc_result != cudaSuccess)
        {
            std::string error_message = "CUDA Error: Failed to malloc gpu memory in ";
            error_message += "File: " + std::string(__FILE__) + " Line: " + std::to_string(__LINE__) + "\n";
            error_message += this->get_matrix_info();
            error_message += "CUDA Error: " + std::string(cudaGetErrorString(malloc_result)) + "\n";
            throw std::runtime_error(error_message);
        }
        this->use_gpu = true;
    }
    template <Arithmetic MatrixDataType>
    void Matrix<MatrixDataType>::cuda() const
    {
        if (!this->use_gpu)
        {
            this->init_cuda();
            this->copy_host_to_device();
        }
    }

    template <Arithmetic MatrixDataType>
    void Matrix<MatrixDataType>::copy_host_to_device() const
    {
        assert(!this->is_external_gpu_device_data); // 如果是外部传入的gpu_device_data，不应该调用这个函数
        cudaError_t cpy_result = cudaMemcpy(this->gpu_device_data, this->data, this->rows_num * this->cols_num * sizeof(MatrixDataType), cudaMemcpyHostToDevice);
        if (cpy_result != cudaSuccess)
        {
            std::string error_message = "CUDA Error: Failed to copy data from host to device in ";
            error_message += "File: " + std::string(__FILE__) + " Line: " + std::to_string(__LINE__) + "\n";
            error_message += this->get_matrix_info();
            error_message += "CUDA Error: " + std::string(cudaGetErrorString(cpy_result)) + "\n";
            throw std::runtime_error(error_message);
        }
    }
    template <Arithmetic MatrixDataType>
    void Matrix<MatrixDataType>::copy_device_to_host()
    {
        cudaError_t cpy_result = cudaMemcpy(this->data, this->gpu_device_data, this->rows_num * this->cols_num * sizeof(MatrixDataType), cudaMemcpyDeviceToHost);
        if (cpy_result != cudaSuccess)
        {
            std::string error_message = "CUDA Error: Failed to copy data from device to host in ";
            error_message += "File: " + std::string(__FILE__) + " Line: " + std::to_string(__LINE__) + "\n";
            error_message += this->get_matrix_info();
            error_message += "CUDA Error: " + std::string(cudaGetErrorString(cpy_result)) + "\n";
            throw std::runtime_error(error_message);
        }
    }

    template <Arithmetic MatrixDataType>
    void Matrix<MatrixDataType>::transpose()
    {
        this->major_axis = (this->major_axis == Axis::Row) ? Axis::Column : Axis::Row;
        uint new_cols_num = this->rows_num;
        this->rows_num = this->cols_num;
        this->cols_num = new_cols_num;
    }

    template <Arithmetic MatrixDataType>
    Matrix<MatrixDataType> Matrix<MatrixDataType>::clone() const
    {
        Matrix<MatrixDataType> matrix = Matrix<MatrixDataType>(*this);
        return matrix;
    }

    template <Arithmetic MatrixDataType>
    Matrix<MatrixDataType> Matrix<MatrixDataType>::get_transpose_matrix() const
    {

        Matrix<MatrixDataType> matrix = this->clone();
        matrix.transpose();
        return matrix;
    }

    template <Arithmetic MatrixDataType>
    Matrix<MatrixDataType> Matrix<MatrixDataType>::get_tmp_transpose_matrix_with_ref_data() const
    { // 一般用于临时变量
        Matrix<MatrixDataType> transposed_matrix =
            Matrix<MatrixDataType>(
                this->data,            // 获取原始的指针
                this->gpu_device_data, // 获取原始的device指针
                this->cols_num,        // 行列互换
                this->rows_num,
                (this->major_axis == Axis::Row) ? Axis::Column : Axis::Row // 行列互换
            );
        // 最好是操作之前先调用cuda,这样get_tmp_transpose_matrix_with_ref_data返回的数据和原来是一致的
        transposed_matrix.use_gpu = this->use_gpu; // 如果原来的矩阵还没有申请gpu的内存，那么新的矩阵在后面的计算需要自己申请
        // 似乎是不需要管use_gpu,如果之前没有use_gpu的话，原来的gpu上的指针会是nullptr
        // 新的矩阵不会影响原来的矩阵
        return transposed_matrix;
    }

    template <Arithmetic MatrixDataType>
    uint Matrix<MatrixDataType>::get_index(uint row, uint col) const
    {
        if (this->major_axis == Axis::Row)
        {
            return row * this->cols_num + col;
        }
        else
        {
            return col * this->rows_num + row;
        }
    }

    template <Arithmetic MatrixDataType>
    MatrixDataType Matrix<MatrixDataType>::get_item(uint row, uint col) const
    {
        return this->data[this->get_index(row, col)];
    }

    template <Arithmetic MatrixDataType>
    std::string Matrix<MatrixDataType>::get_matrix_info() const
    {
        std::string matrix_info = "Matrix Info:\n";
        matrix_info += "  Rows: " + std::to_string(this->rows_num) + " Columns: " + std::to_string(this->cols_num) + "\n";
        matrix_info += "  Major Axis: " + std::string(this->major_axis == Axis::Row ? "Row" : "Column") + "\n";
        matrix_info += "  Data From External: " + std::string(this->is_external_data ? "True" : "False") + "\n";
        matrix_info += "  GPU Data From External: " + std::string(this->is_external_gpu_device_data ? "True" : "False") + "\n";
        return matrix_info;
    }

    template <Arithmetic MatrixDataType>
    void Matrix<MatrixDataType>::show() const
    {
        std::cout << "The matrix with " << this->rows_num << " rows and " << this->cols_num << " columns is as follows:" << std::endl;
        for (uint row = 0; row < this->rows_num; row++)
        {
            for (uint col = 0; col < this->cols_num; col++)
            {
                std::cout << this->get_item(row, col) << " ";
            }
            std::cout << std::endl;
        }
    }

    template <Arithmetic MatrixDataType>
    Matrix<MatrixDataType> Matrix<MatrixDataType>::zeros(uint rows_num, uint cols_num)
    {
        Matrix<MatrixDataType> matrix = Matrix<MatrixDataType>();
        matrix.rows_num = rows_num;
        matrix.cols_num = cols_num;
        matrix.data = new MatrixDataType[rows_num * cols_num](); // 分配内存并初始化为0
        return matrix;
    }

    template <Arithmetic OutputMatrixDataType>
    template <Arithmetic TemplateMatrixDataType>
    Matrix<OutputMatrixDataType> Matrix<OutputMatrixDataType>::zeros_like(
        const Matrix<TemplateMatrixDataType> &matrix)
    {
        return matrix.template zeros_like<OutputMatrixDataType>();
    }

    template <Arithmetic TemplateMatrixDataType> // 类模板参数,是模板矩阵的数据类型
    template <Arithmetic OutputMatrixDataType>   // 函数模板参数,是输出矩阵的数据类型
    Matrix<OutputMatrixDataType> Matrix<TemplateMatrixDataType>::zeros_like() const
    {
        return Matrix<OutputMatrixDataType>::zeros(this->rows_num, this->cols_num);
    }

    template <Arithmetic MatrixDataType>
    Matrix<MatrixDataType> Matrix<MatrixDataType>::ones(uint rows_num, uint cols_num)
    {
        Matrix<MatrixDataType> matrix = Matrix<MatrixDataType>::zeros(rows_num, cols_num);
        std::fill(matrix.data, matrix.data + rows_num * cols_num, static_cast<MatrixDataType>(1));
        return matrix;
    }

    template <Arithmetic MatrixDataType>
    template <Arithmetic OutputMatrixDataType>
    Matrix<OutputMatrixDataType> Matrix<MatrixDataType>::ones_like(
        const Matrix<MatrixDataType> &matrix)
    {
        return Matrix<OutputMatrixDataType>::ones(matrix.rows_num, matrix.cols_num);
    }

    template <Arithmetic MatrixDataType>
    Matrix<MatrixDataType> Matrix<MatrixDataType>::multiply(
        const Matrix<MatrixDataType> &matrix) const
    {
        Matrix<MatrixDataType> output = Matrix<MatrixDataType>::zeros(this->rows_num, matrix.cols_num);
        this->multiply(matrix, output);
        return output;
    }

    template <Arithmetic MatrixDataType>
    void Matrix<MatrixDataType>::multiply(
        const Matrix<MatrixDataType> &input,
        Matrix<MatrixDataType> &output) const
    {
        // 要算的是output=this*input
        assert(this->cols_num == input.rows_num);
        assert(this->rows_num == output.rows_num);
        assert(input.cols_num == output.cols_num);
        // 暂时要求所有矩阵都是行优先的
        // 如果有一个矩阵不是行优先的，直接抛出未定义异常
        // if (this->major_axis != Axis::Row || input.major_axis != Axis::Row || output.major_axis != Axis::Row)
        // {
        //     throw std::runtime_error("The matrix is not row-major matrix");
        // }
        // 要求MatrixDataType是float,否则抛出未定义异常
        if (!std::is_same<MatrixDataType, float>::value)
        {
            throw std::runtime_error("The matrix data type is not float");
        }

        // // 调用cuda的矩阵乘法
        // float *d_this, *d_input, *d_output;
        // cudaMalloc(&d_this, this->rows_num * this->cols_num * sizeof(float));
        // cudaMalloc(&d_input, input.rows_num * input.cols_num * sizeof(float));
        // cudaMalloc(&d_output, output.rows_num * output.cols_num * sizeof(float));
        // // 先将数据复制到CUDA上面
        // cudaMemcpy(d_this, this->data, sizeof(float) * this->rows_num * this->cols_num, cudaMemcpyHostToDevice);
        // cudaMemcpy(d_input, input.data, sizeof(float) * input.rows_num * input.cols_num, cudaMemcpyHostToDevice);
        this->cuda();
        input.cuda();
        output.cuda();

        const uint BK = 8;
        const uint TM = 8;
        const uint TN = 8;
        if (this->rows_num >= 128 and input.cols_num >= 128)
        {
            const uint BM = 128;
            const uint BN = 128;
            dim3 gridDim(CEIL_DIV(input.cols_num, BN), CEIL_DIV(this->rows_num, BM));
            dim3 blockDim((BM * BN) / (TM * TN));
            if (this->major_axis == Axis::Row)
            {

                CUDA_KERNELS::matmul_row_major<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(this->rows_num, input.cols_num, this->cols_num, 1.0, this->gpu_device_data, input.gpu_device_data, 0.0, output.gpu_device_data);
            }
            else
            {
                CUDA_KERNELS::matmul_with_A_col_major_and_B_C_row_major<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(this->rows_num, input.cols_num, this->cols_num, 1.0, this->gpu_device_data, input.gpu_device_data, 0.0, output.gpu_device_data);
            }
        }
        else
        {
            const uint BM = 64;
            const uint BN = 64;
            dim3 gridDim(CEIL_DIV(input.cols_num, BN), CEIL_DIV(this->rows_num, BM));
            dim3 blockDim((BM * BN) / (TM * TN));
            if (this->major_axis == Axis::Row)
            {

                CUDA_KERNELS::matmul_row_major<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(this->rows_num, input.cols_num, this->cols_num, 1.0, this->gpu_device_data, input.gpu_device_data, 0.0, output.gpu_device_data);
            }
            else
            {
                CUDA_KERNELS::matmul_with_A_col_major_and_B_C_row_major<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(this->rows_num, input.cols_num, this->cols_num, 1.0, this->gpu_device_data, input.gpu_device_data, 0.0, output.gpu_device_data);
            }
        }

        // 将结果复制回来
        output.copy_device_to_host();
    }

    template <Arithmetic MatrixDataType>
    void Matrix<MatrixDataType>::multiply(
        const float scale,
        Matrix<MatrixDataType> &output) const
    {
        if (!std::is_same<MatrixDataType, float>::value)
        {
            throw std::runtime_error("The matrix data type is not float");
        }

        this->cuda();
        output.cuda();

        dim3 blockDim(16, 16);
        dim3 gridDim(CEIL_DIV(this->cols_num, blockDim.x), CEIL_DIV(this->rows_num, blockDim.y));
        CUDA_KERNELS::matmul_by_scale_same_major_axis<<<gridDim, blockDim>>>(this->gpu_device_data, this->rows_num, this->cols_num, scale, output.gpu_device_data);

        // 将结果复制回来
        output.copy_device_to_host();
    }

    template <Arithmetic MatrixDataType>
    void Matrix<MatrixDataType>::add(
        const Matrix<MatrixDataType> &input,
        Matrix<MatrixDataType> &output) const
    {
        if (this->rows_num != input.rows_num || this->rows_num != output.rows_num || this->cols_num != input.cols_num || this->cols_num != output.cols_num)
        {
            throw std::runtime_error("The matrix shape is not the same");
        }

        if (!std::is_same<MatrixDataType, float>::value)
        {
            throw std::runtime_error("The matrix data type is not float");
        }

        this->cuda();
        input.cuda();
        output.cuda();

        dim3 blockDim(16, 16);
        dim3 gridDim(CEIL_DIV(this->cols_num, blockDim.x), CEIL_DIV(this->rows_num, blockDim.y));
        CUDA_KERNELS::matadd_row_major<<<gridDim, blockDim>>>(this->gpu_device_data, input.gpu_device_data, this->rows_num, this->cols_num, output.gpu_device_data);

        // 将结果复制回来
        output.copy_device_to_host();
    }

    template <Arithmetic MatrixDataType>
    void Matrix<MatrixDataType>::add_assign(
        const Matrix<MatrixDataType> &matrix)
    {
        // 要求输入matrix与当前matrix行数一致
        if (this->rows_num != matrix.rows_num)
        {
            throw std::runtime_error("The matrix shape is not the same");
        }
        else if (this->cols_num != matrix.cols_num && matrix.cols_num != 1)
        {
            throw std::runtime_error("The matrix shape is not the same");
        }
        else if (this->cols_num != matrix.cols_num && matrix.cols_num == 1)
        {
            this->cuda();
            matrix.cuda();

            dim3 blockDim(16, 16);
            dim3 gridDim(CEIL_DIV(this->cols_num, blockDim.x), CEIL_DIV(this->rows_num, blockDim.y));
            CUDA_KERNELS::matadd_assign_bias_row_major<<<gridDim, blockDim>>>(this->gpu_device_data, matrix.gpu_device_data, this->rows_num, this->cols_num);
            this->copy_device_to_host();
        }
        else if (this->cols_num == matrix.cols_num && this->rows_num == matrix.rows_num)
        {
            this->cuda();
            matrix.cuda();

            dim3 blockDim(16, 16);
            dim3 gridDim(CEIL_DIV(this->cols_num, blockDim.x), CEIL_DIV(this->rows_num, blockDim.y));
            CUDA_KERNELS::matadd_assign_row_major<<<gridDim, blockDim>>>(this->gpu_device_data, matrix.gpu_device_data, this->rows_num, this->cols_num);
            this->copy_device_to_host();
        }
        else
        {
            throw std::runtime_error("The matrix shape is not the same");
        }
    }

    template <Arithmetic MatrixDataType>
    Matrix<MatrixDataType> Matrix<MatrixDataType>::add(
        const Matrix<MatrixDataType> &matrix) const
    {
        Matrix<MatrixDataType> output = Matrix<MatrixDataType>::zeros(this->rows_num, matrix.cols_num);
        this->add(matrix, output);
        return output;
    }

    template <Arithmetic MatrixDataType>
    void Matrix<MatrixDataType>::sum(
        Matrix<MatrixDataType> &output, Axis axis) const
    {
        this->cuda();
        output.cuda();
        dim3 blockDim(16, 16);
        if (axis == Axis::Row)
        {
            dim3 gridDim(CEIL_DIV(output.cols_num, blockDim.x * blockDim.y));
            CUDA_KERNELS::sum_row_major_by_row<<<gridDim, blockDim>>>(this->gpu_device_data, this->rows_num, this->cols_num, output.gpu_device_data);
        }
        else
        {
            dim3 gridDim(CEIL_DIV(output.rows_num, blockDim.x * blockDim.y));
            CUDA_KERNELS::sum_row_major_by_col<<<gridDim, blockDim>>>(this->gpu_device_data, this->rows_num, this->cols_num, output.gpu_device_data);
        }
        output.copy_device_to_host();
    }

    template <Arithmetic MatrixDataType>
    void Matrix<MatrixDataType>::relu(
        Matrix<MatrixDataType> &output) const
    {
        this->cuda();
        output.cuda();
        dim3 blockDim(16, 16);
        dim3 gridDim(CEIL_DIV(this->cols_num, blockDim.x), CEIL_DIV(this->rows_num, blockDim.y));
        CUDA_KERNELS::mat_relu<<<gridDim, blockDim>>>(this->gpu_device_data, this->rows_num * this->cols_num, output.gpu_device_data);
        output.copy_device_to_host();
    }

    template <Arithmetic MatrixDataType>
    void Matrix<MatrixDataType>::relu()
    {
        this->cuda();
        dim3 blockDim(16, 16);
        dim3 gridDim(CEIL_DIV(this->cols_num, blockDim.x), CEIL_DIV(this->rows_num, blockDim.y));
        CUDA_KERNELS::mat_relu<<<gridDim, blockDim>>>(this->gpu_device_data, this->rows_num * this->cols_num, this->gpu_device_data);
        this->copy_device_to_host();
    }

    template <Arithmetic MatrixDataType>
    Matrix<MatrixDataType> Matrix<MatrixDataType>::get_relu_matrix() const
    {
        Matrix<MatrixDataType> output = this->clone();
        this->relu(output);
        return output;
    }
}

namespace DataSet
{
    class MnistData
    {
    private:
        std::string images_file_path;
        std::string labels_file_path;
        uint num_elements;
        float *images_data;
        uint *labels_data;

    public:
        MnistData(std::string images_file_path,
                  std::string labels_file_path,
                  uint num_elements);
        ~MnistData();

        void load_images_data();
    };

}

namespace NeuralNetwork
{
    class Linear
    {
    public:
        const uint input_features;
        const uint output_features;
        Math::Matrix<float> weights;
        Math::Matrix<float> bias;
        Math::Matrix<float> grad_weights;
        Math::Matrix<float> grad_bias;
        Math::Matrix<float> grad_input;

    public:
        Linear(uint input_features, uint output_features);
        ~Linear();

        void initialize_weights();
        void initialize_bias();
        void initialize_parameters();

        void forward(const Math::Matrix<float> &input, Math::Matrix<float> &output) const;
        Math::Matrix<float> forward(const Math::Matrix<float> &input) const;

        void backward(const Math::Matrix<float> &grad_output, const Math::Matrix<float> &input);

        void update_weights(float learning_rate);
    };

    class ReLU
    {
    private:
        /* data */
    public:
        ReLU(/* args */);
        ~ReLU();
        void forward(const Math::Matrix<float> &input, Math::Matrix<float> &output) const;
        void forward(Math::Matrix<float> &input) const;
        // Math::Matrix<float> forward(const Math::Matrix<float> &input) const;

        void backward(const Math::Matrix<float> &grad_output, const Math::Matrix<float> &input, Math::Matrix<float> &grad_input);
    };

    class SoftMax
    {
    private:
        /* data */
    public:
        SoftMax(/* args */) = default;
        ~SoftMax() = default;
        void forward(const Math::Matrix<float> &input, Math::Matrix<float> &output) const;
        void forward(Math::Matrix<float> &input) const;

        void backward(const Math::Matrix<float> &grad_output, const Math::Matrix<float> &input, Math::Matrix<float> &grad_input);
    };

    class CrossEntropyLoss
    {
    private:
        SoftMax softmax;

    public:
        CrossEntropyLoss(/* args */) = default;
        ~CrossEntropyLoss() = default;

        // input[10*batch_size],target[batch_size*1]
        float forward(Math::Matrix<float> &input, Math::Matrix<int> &target) const;
    };

    class MLP
    {
    private:
        uint input_features;
        uint hidden_features;
        uint num_classes;
        Linear fc1;
        ReLU relu;
        Linear fc2;
        SoftMax softmax;
        CrossEntropyLoss cross_entropy_loss;
        Math::Matrix<float> fc1_output;
        Math::Matrix<float> relu_output;
        Math::Matrix<float> fc2_output;

    public:
        MLP(uint input_features, uint hidden_features, uint num_classes);
        ~MLP();
        void forward(const Math::Matrix<float> &input);
        void backward(const Math::Matrix<float> &grad_output);
        void update_weights(float learning_rate);

        void train(DataSet::MnistData &train_data, float learning_rate, uint epochs);
    };

} // namespace NeuralNetwork

namespace NeuralNetwork
{
    Linear::Linear(uint input_features,
                   uint output_features)
        : input_features(input_features),
          output_features(output_features),
          weights(Math::Matrix<float>::zeros(output_features, input_features)),
          bias(Math::Matrix<float>::zeros(output_features, 1)),
          grad_weights(Math::Matrix<float>::zeros(output_features, input_features)),
          grad_bias(Math::Matrix<float>::zeros(output_features, 1)),
          grad_input(Math::Matrix<float>::zeros(input_features, 1))
    {
        this->initialize_parameters();
    }

    Linear::~Linear()
    {
    }

    void Linear::initialize_weights()
    {
        const float scale = sqrtf(2.0 / (this->input_features));
        for (uint i = 0; i < (this->weights.rows_num) * (this->weights.cols_num); i++)
        {
            this->weights.data[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
        }
    }
    void Linear::initialize_bias()
    {
        // 就是0,不用变
    }
    void Linear::initialize_parameters()
    {
        this->initialize_weights();
        this->initialize_bias();
    }

    Math::Matrix<float> Linear::forward(const Math::Matrix<float> &input) const
    {
        Math::Matrix<float> output = Math::Matrix<float>::zeros(this->output_features, input.cols_num);
        this->forward(input, output);
        return output;
    }

    void Linear::forward(const Math::Matrix<float> &input, Math::Matrix<float> &output) const
    {
        // output = weights * input + bias
        this->weights.multiply(input, output);
        output.add_assign(this->bias);
    }

    void Linear::backward(const Math::Matrix<float> &grad_output, const Math::Matrix<float> &input)
    {
        // grad_weights=grad_output*input^T
        grad_output.multiply(input.get_tmp_transpose_matrix_with_ref_data(), this->grad_weights);

        grad_output.sum(this->grad_bias, Math::Axis::Column);

        // grad_input=weights^T*grad_output
        this->weights.get_tmp_transpose_matrix_with_ref_data().multiply(grad_output, this->grad_input);
    }
    void Linear::update_weights(float learning_rate)
    {
        this->grad_weights.multiply(-learning_rate, this->grad_weights);
        this->weights.add_assign(this->grad_weights);
        this->grad_bias.multiply(-learning_rate, this->grad_bias);
        this->bias.add_assign(this->grad_bias);
    }

} // namespace NeuralNetwork

namespace NeuralNetwork
{
    ReLU::ReLU(/* args */)
    {
    }

    ReLU::~ReLU()
    {
    }

    void ReLU::forward(const Math::Matrix<float> &input, Math::Matrix<float> &output) const
    {
        input.relu(output);
    }

    void ReLU::forward(Math::Matrix<float> &input) const
    {
        input.relu();
    }

    void ReLU::backward(const Math::Matrix<float> &grad_output, const Math::Matrix<float> &input, Math::Matrix<float> &grad_input)
    {
        dim3 blockDim(16, 16);
        dim3 gridDim(CEIL_DIV(input.cols_num, blockDim.x), CEIL_DIV(input.rows_num, blockDim.y));
        // 要求三个的维度和优先维度一致
        assert(input.rows_num == grad_output.rows_num && input.cols_num == grad_output.cols_num);
        assert(input.rows_num == grad_input.rows_num && input.cols_num == grad_input.cols_num);
        assert(input.major_axis == grad_output.major_axis && input.major_axis == grad_input.major_axis);
        input.cuda();
        grad_output.cuda();
        grad_input.cuda();
        CUDA_KERNELS::mat_relu_backward<<<gridDim, blockDim>>>(input.gpu_device_data, input.rows_num * input.cols_num, grad_output.gpu_device_data, grad_input.gpu_device_data);
        grad_input.copy_device_to_host();
    }
}

namespace NeuralNetwork
{
    void SoftMax::forward(const Math::Matrix<float> &input, Math::Matrix<float> &output) const
    {
        assert(input.major_axis == Math::Axis::Row);
        assert(output.major_axis == Math::Axis::Row);
        // 维度相等
        assert(input.rows_num == output.rows_num && input.cols_num == output.cols_num);
        input.cuda();
        output.cuda();
        dim3 blockDim(16, 16);
        dim3 gridDim(CEIL_DIV(input.cols_num, blockDim.x), CEIL_DIV(input.rows_num, blockDim.y));
        CUDA_KERNELS::softmax_row_major<<<gridDim, blockDim>>>(input.gpu_device_data, input.rows_num, input.cols_num, output.gpu_device_data);
        output.copy_device_to_host();
    }

    void SoftMax::forward(Math::Matrix<float> &input) const
    {
        assert(input.major_axis == Math::Axis::Row);
        input.cuda();
        dim3 blockDim(16, 16);
        dim3 gridDim(CEIL_DIV(input.cols_num, blockDim.x), CEIL_DIV(input.rows_num, blockDim.y));
        CUDA_KERNELS::softmax_row_major<<<gridDim, blockDim>>>(input.gpu_device_data, input.rows_num, input.cols_num, input.gpu_device_data);
        input.copy_device_to_host();
    }
} // namespace NeuralNetwork

namespace NeuralNetwork
{
    float CrossEntropyLoss::forward(Math::Matrix<float> &input, Math::Matrix<int> &target) const
    {
        Math::Matrix<float> softmax_output = Math::Matrix<float>::zeros_like(input);
        this->softmax.forward(input, softmax_output);
        Math::Matrix<float> loss_matrix = Math::Matrix<float>::zeros(1, 1);
        loss_matrix.init_cuda();
        loss_matrix.cuda();
        target.cuda();
        softmax_output.cuda();
        dim3 blockDim(16, 16);
        dim3 gridDim(CEIL_DIV(input.cols_num, blockDim.x));
        CUDA_KERNELS::cross_entropy_loss_row_major<<<gridDim, blockDim>>>(softmax_output.gpu_device_data, softmax_output.rows_num, softmax_output.cols_num, target.gpu_device_data, loss_matrix.gpu_device_data);
        loss_matrix.copy_device_to_host();
        return loss_matrix.data[0];
    }
} // namespace NeuralNetwork

namespace NeuralNetwork
{

    MLP::MLP(uint input_features, uint hidden_features, uint num_classes)
        : input_features(input_features),
          hidden_features(hidden_features),
          num_classes(num_classes),
          fc1(Linear(input_features, hidden_features)),
          relu(ReLU()),
          fc2(Linear(hidden_features, num_classes)),
          softmax(SoftMax()),
          cross_entropy_loss(CrossEntropyLoss()),
          fc1_output(Math::Matrix<float>::zeros(hidden_features, 1)),
          relu_output(Math::Matrix<float>::zeros(hidden_features, 1)),
          fc2_output(Math::Matrix<float>::zeros(num_classes, 1))
    {
    }

    void MLP::forward(const Math::Matrix<float> &input)
    {
        this->fc1.forward(input, this->fc1_output);
        this->relu.forward(this->fc1_output);
        this->fc2.forward(this->fc1_output, this->fc2_output);
        this->softmax.forward(this->fc2_output);
    }

    void MLP::backward(const Math::Matrix<float> &grad_output)
    {
        this->fc2.backward(grad_output, this->relu_output);
        Math::Matrix<float> grad_relu_input = Math::Matrix<float>::zeros_like(this->fc1_output);
        grad_relu_input.init_cuda();
        grad_relu_input.cuda();
        this->relu.backward(this->fc2.grad_input, this->fc1_output, grad_relu_input);
        this->fc1.backward(grad_relu_input, this->fc1_output);
    }
} // namespace NeuralNetwork

namespace DataSet
{
    // impl MnistData
    MnistData::MnistData(std::string images_file_path,
                         std::string labels_file_path,
                         uint num_elements)
    {
        this->images_file_path = images_file_path;
        this->labels_file_path = labels_file_path;
        this->num_elements = num_elements;
        this->images_data = new float[num_elements * CONSTANTS::PICTURE_SIZE];
        this->labels_data = new uint[num_elements * CONSTANTS::LABEL_SIZE];
    }
    MnistData::~MnistData()
    {
        delete[] this->images_data;
        delete[] this->labels_data;
    }
}
int main()
{
    // using namespace Math;
    // Matrix m = Matrix<long>::zeros(3, 4);
    // std::cout << sizeof(m.get_item(0, 0)) << std::endl;
    // m.show();

    // Matrix zeros_float = Matrix<>::zeros_like(m);
    // std::cout << sizeof(zeros_float.get_item(0, 0)) << std::endl;
    // zeros_float.show();

    // Matrix zeros_bit = Matrix<bool>::zeros_like(m);
    // std::cout << sizeof(zeros_bit.get_item(0, 0)) << std::endl;
    // zeros_bit.show();
    // zeros_bit.get_transpose_matrix().show();

    using namespace NeuralNetwork;
    Linear linear = Linear(3, 4);
    linear.weights.show();
    const uint batch_size = 4;
    Math::Matrix<float> input = Math::Matrix<float>::ones(3, batch_size);
    Math::Matrix<float> output = linear.forward(input);
    output.show();

    ReLU relu = ReLU();
    relu.forward(output);
    output.show();
}