#include <stdio.h>
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
        const uint threadRow = threadIdx.x / small_C_dim.x;
        const uint threadCol = threadIdx.x % small_C_dim.x;

        return;
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

        Matrix<MatrixDataType> mutiply(const Matrix<MatrixDataType> &matrix) const;
        void mutiply(const Matrix<MatrixDataType> &input, Matrix<MatrixDataType> &output);
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
    Matrix<MatrixDataType> Matrix<MatrixDataType>::mutiply(
        const Matrix<MatrixDataType> &matrix) const
    {
        Matrix<MatrixDataType> output = Matrix<MatrixDataType>::zeros(this->rows_num, matrix.cols_num);
        this->mutiply(matrix, output);
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