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
#define VECTOR_SIZE 4
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
        // 处理边界条件，考虑使用向量化加速

        // i\in [0,I), M=I\times BM, 是分块矩阵A和C的行数
        // j\in [0,J), N=J\times BN, 是分块矩阵B和C的列数
        const uint C_row_index_i = blockIdx.y; // i
        const uint C_col_index_j = blockIdx.x; // j
        // K=U\times BK,是分块后小矩阵A_{m_i}再次分块的列数，也是分块后小矩阵B_{n_j}再次分块的行数
        // 分块后小矩阵C_{m_in_j}需要U个A_{m_ik_u}*B_{k_un_j}相加
        const uint U = CEIL_DIV(K, BK);

        // BM=T\times TM, BN=S\times TN
        // T是As分块后的行数，S是Bs分块后的列数
        // T和S也是小矩阵Cs分块后的行数和列数
        const dim3 Cs_dim = {CEIL_DIV(BN, TN), CEIL_DIV(BM, TM)}; // S列,T行
        // t是当前线程负责的Cs分块后的小矩阵的行下标
        const uint Cs_row_t = threadIdx.x / small_C_dim.x; //(Idx)/S\in [0,T)
        // s是当前线程负责的Cs分块后的小矩阵的列下标
        const uint Cs_col_s = threadIdx.x % small_C_dim.x; //(Idx)%S\in [0,S)

        // 当前线程负责计算的TM*TN个元素
        float Cs_m_t_n_s[TM * TN] = {0.0};

        // 当前block计算时需要的分块矩阵数据
        __shared__ float As[BM * BK];
        __shared__ float Bs[BK * BN];

        // 移动指针到当前block需要计算的C_{m_in_j}的起始位置
        A += C_row_index_i * BM * K;                      // 向下移动i行，每一个小的分块矩阵的行数是BM，大矩阵每一行有K个元素
        B += C_col_index_j * BK;                          // 向右移动j列，每一个小的分块矩阵的列数是BK
        C += C_row_index_i * BM * N + C_col_index_j * BN; // 向右移动j列，每一个小的分块矩阵的列数是BN，向下移动i行，每一个小的分块矩阵的行数是BM,大矩阵每一行有N个元素

        // 下面是给每个线程分配加载数据的任务
        const uint As_row = threadIdx.x / (BK / VECTOR_SIZE); // 最大为VECTOR_SIZE*T*S/(B*K)-1
        const uint As_col = threadIdx.x % (BK / VECTOR_SIZE); // 最大为B*K/VECTOR_SIZE-1
        // 需要重复加载的次数
        const uint strideA = BM * BK / (Cs_dim.x * Cs_dim.y); // 数据总数处以线程总数

        const uint Bs_row = threadIdx.x / (BN / VECTOR_SIZE);
        const uint Bs_col = threadIdx.x % (BN / VECTOR_SIZE);
        const uint strideB = BK * BN / (Cs_dim.x * Cs_dim.y);

        // 外层循环，循环U次，每次计算一个小矩阵C_{m_in_j}（对于整个block来说，对于这个线程来说是计算这个小矩阵中的一个小分块）
        for (uint u = 0; u < U; u++)
        {
            // 将这个block计算需要的As和Bs加载到共享内存中
            for (uint loadOffset = 0; loadOffset < strideA; loadOffset++)
            {
                // 获取As的第As_row行，第As_col列到第(As_col+3)列的数据
                float4 tmp = reinterpret_cast<float4 *>(&A[(As_row + loadOffset) * K + As_col * VECTOR_SIZE])[0];
                // 将数据转置后加载到共享内存中
                As[(As_col * VECTOR_SIZE + 0) * BM + As_row + loadOffset] = tmp.x;
                As[(As_col * VECTOR_SIZE + 1) * BM + As_row + loadOffset] = tmp.y;
                As[(As_col * VECTOR_SIZE + 2) * BM + As_row + loadOffset] = tmp.z;
                As[(As_col * VECTOR_SIZE + 3) * BM + As_row + loadOffset] = tmp.w;
            }
            for (uint loadOffset = 0; loadOffset < strideB; loadOffset++)
            {
                // 获取Bs的第Bs_row行，第Bs_col列到第(Bs_col+3)列的数据
                reinterpret_cast<float4 *>(&Bs[(Bs_row + loadOffset) * BN + Bs_col * VECTOR_SIZE])[0] =
                    reinterpret_cast<float4 *>(&B[(Bs_row + loadOffset) * N + Bs_col * VECTOR_SIZE])[0];
            }
            __syncthreads();
            A += BK;     // A向右移动BK
            B += BK * N; // B向下移动BK

            // 来到当前线程要计算的Cs_{m_tn_s}=As_{m_t}Bs_{n_s}
            // As_{m_t}：TM*BK, Bs_{n_s}:BK*TN, Cs_{m_tn_s}:TM*TN
            // for (uint _As_m_t_row = 0; _As_m_t_row < TM; _As_m_t_row++)
            // {
            //     for (uint _Bs_n_s = 0; _Bs_n_s < TN; _Bs_n_s++)
            //     {
            //         for (uint _dot_index = 0; _dot_index < BK; _dot_index++)
            //         {
            //             Cs_m_t_n_s[_As_m_t_row * TN + _Bs_n_s] +=
            //                 As[_dot_index * BM + Cs_row_t * TM + _As_m_t_row] * // As是转置的
            //                 Bs[_dot_index * BN + _Bs_n_s + Cs_col_s * TN];
            //         }
            //     }
            // }

            // 根据上面的逻辑进行改写优化
            for (uint _dot_index = 0; _dot_index < BK; _dot_index++)
            {
                for (uint _As_m_t_row = 0; _As_m_t_row < TM; _As_m_t_row++)
                {
                    for (uint _Bs_n_s = 0; _Bs_n_s < TN; _Bs_n_s++)
                    {
                        Cs_m_t_n_s[_As_m_t_row * TN + _Bs_n_s] +=
                            As[_dot_index * BM + Cs_row_t * TM + _As_m_t_row] * // As是转置的
                            Bs[_dot_index * BN + _Bs_n_s + Cs_col_s * TN];
                    }
                }
            }
            __syncthreads();
        }

        // 将计算结果写回去
        for (uint _res_row = 0; _res_row < TM; _res_row++)
        {
            for (uint _res_col = 0; _res_col < TN; _res_col += VECTOR_SIZE)
            {
                float4 tmp = reinterpret_cast<float4 *>(&C[(Cs_row_t * TM + _res_row) * N + Cs_col_s * TN + _res_col])[0];
                tmp.x = alpha * Cs_m_t_n_s[_res_row * TN + _res_col] + beta * tmp.x;
                tmp.y = alpha * Cs_m_t_n_s[_res_row * TN + _res_col + 1] + beta * tmp.y;
                tmp.z = alpha * Cs_m_t_n_s[_res_row * TN + _res_col + 2] + beta * tmp.z;
                tmp.w = alpha * Cs_m_t_n_s[_res_row * TN + _res_col + 3] + beta * tmp.w;

                reinterpret_cast<float4 *>(&C[(Cs_row_t * TM + _res_row) * N + Cs_col_s * TN + _res_col])[0] = tmp;
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
    private:
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
        void multiply(const Matrix<MatrixDataType> &input, Matrix<MatrixDataType> &output);
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
        Matrix<MatrixDataType> &output)
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
        float *d_this, d_input, d_output;
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
            dim3 blockDim((BM * BN) / TM * TN);
            CUDA_KERNELS::matmul<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(this->rows_num, input.cols_num, this->cols_num, 1.0, d_this, d_input, 0.0, d_output);
        }
        else
        {
            const uint BM = 64;
            const uint BN = 64;
            dim3 gridDim(CEIL_DIV(input.cols_num, BN), CEIL_DIV(this->rows_num, BM));
            dim3 blockDim((BM * BN) / TM * TN);
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

int main()
{
    using namespace Math;
    Matrix m = Matrix<long>::zeros(3, 4);
    std::cout << sizeof(m.get_item(0, 0)) << std::endl;
    m.show();

    Matrix zeros_float = Matrix<>::zeros_like(m);
    std::cout << sizeof(zeros_float.get_item(0, 0)) << std::endl;
    zeros_float.show();

    Matrix zeros_bit = Matrix<bool>::zeros_like(m);
    std::cout << sizeof(zeros_bit.get_item(0, 0)) << std::endl;
    zeros_bit.show();
    zeros_bit.get_transpose_matrix().show();
}