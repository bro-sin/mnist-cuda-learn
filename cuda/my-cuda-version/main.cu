#include <stdio.h>
#include <cassert>
#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include <concepts>

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
    template <const uint BM, const uint BN, const uint BK, const uint TM, const uint TN>
    __global__ void matmul(
        const int M,
        const int N,
        const int K,
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
    __global__ void matadd(const float *A,
                           const float *B,
                           const int m,
                           const int n,
                           float *C)
    {
        return;
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
        { a *b } -> std::convertible_to<T>;
        { a / b } -> std::convertible_to<T>;
    };

    template <Arithmetic MatrixDataType = float>
    class Matrix
    {
    public:
        MatrixDataType *data;
        uint rows_num;
        uint cols_num;
        const bool is_external_data = false; // 用于判断data是外界传入的还是内部创建的，默认数据是内部创建的，应当在析构函数中释放内存
        Axis major_axis = Axis::Row;

        Matrix(); // 完全用于内部创建对象

        uint get_index(uint row, uint col) const;

    public:
        Matrix(MatrixDataType *data,
               uint rows_num,
               uint cols_num,
               Axis major_axis = Axis::Row); // 给外面调用的构造函数，保持is_external_data为true
        Matrix(const Matrix<MatrixDataType> &matrix);
        ~Matrix();

        void show() const;

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

        Matrix<MatrixDataType> multiply(const Matrix<MatrixDataType> &matrix) const;
        void multiply(const Matrix<MatrixDataType> &input, Matrix<MatrixDataType> &output) const;
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
        if (this->major_axis != Axis::Row || input.major_axis != Axis::Row || output.major_axis != Axis::Row)
        {
            throw std::runtime_error("The matrix is not row-major matrix");
        }
        // 要求MatrixDataType是float,否则抛出未定义异常
        if (!std::is_same<MatrixDataType, float>::value)
        {
            throw std::runtime_error("The matrix data type is not float");
        }

        // 调用cuda的矩阵乘法
        float *d_this, *d_input, *d_output;
        cudaMalloc(&d_this, this->rows_num * this->cols_num * sizeof(float));
        cudaMalloc(&d_input, input.rows_num * input.cols_num * sizeof(float));
        cudaMalloc(&d_output, output.rows_num * output.cols_num * sizeof(float));
        // 先将数据复制到CUDA上面
        cudaMemcpy(d_this, this->data, sizeof(float) * this->rows_num * this->cols_num, cudaMemcpyHostToDevice);
        cudaMemcpy(d_input, input.data, sizeof(float) * input.rows_num * input.cols_num, cudaMemcpyHostToDevice);

        const uint BK = 8;
        const uint TM = 8;
        const uint TN = 8;
        if (this->rows_num >= 128 and input.cols_num >= 128)
        {
            const uint BM = 128;
            const uint BN = 128;
            dim3 gridDim(CEIL_DIV(input.cols_num, BN), CEIL_DIV(this->rows_num, BM));
            dim3 blockDim((BM * BN) / (TM * TN));
            CUDA_KERNELS::matmul<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(this->rows_num, input.cols_num, this->cols_num, 1.0, d_this, d_input, 0.0, d_output);
        }
        else
        {
            const uint BM = 64;
            const uint BN = 64;
            dim3 gridDim(CEIL_DIV(input.cols_num, BN), CEIL_DIV(this->rows_num, BM));
            dim3 blockDim((BM * BN) / (TM * TN));
            CUDA_KERNELS::matmul<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(this->rows_num, input.cols_num, this->cols_num, 1.0, d_this, d_input, 0.0, d_output);
        }

        // 将结果复制回来
        cudaMemcpy(output.data, d_output, sizeof(float) * output.rows_num * output.cols_num, cudaMemcpyDeviceToHost);
        // 释放内存
        cudaFree(d_this);
        cudaFree(d_input);
        cudaFree(d_output);
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

// int main()
// {
//     using namespace Math;
//     Matrix m = Matrix<long>::zeros(3, 4);
//     std::cout << sizeof(m.get_item(0, 0)) << std::endl;
//     m.show();

//     Matrix zeros_float = Matrix<>::zeros_like(m);
//     std::cout << sizeof(zeros_float.get_item(0, 0)) << std::endl;
//     zeros_float.show();

//     Matrix zeros_bit = Matrix<bool>::zeros_like(m);
//     std::cout << sizeof(zeros_bit.get_item(0, 0)) << std::endl;
//     zeros_bit.show();
//     zeros_bit.get_transpose_matrix().show();
// }