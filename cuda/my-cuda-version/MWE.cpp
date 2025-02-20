// g++ MWE.cpp
#include <iostream>
template <typename T>
class Matrix
{
public:
    Matrix();
    ~Matrix();

    template <typename OutT = T>
    Matrix<OutT> zeros_like() const;

    template <typename T1>
    static void foo1(const Matrix<T> &mat1)
    { // foo1可以正常编译
        Matrix zeros1_like_input = mat1.zeros_like<T>();
        Matrix zeros1_using_template = mat1.zeros_like<T1>();
    }

    template <typename T2>
    static void foo2(const Matrix<T2> &mat2)
    { // foo2会报错：
        // expected primary-expression before ‘>’ token
        // expected primary-expression before ‘)’ token
        Matrix zeros2_using_class_template = mat2.zeros_like<T>();
        Matrix zeros2_like_input = mat2.zeros_like<T2>();
        // 改成下面版本可以正常编译
        // Matrix zeros2_using_class_template = mat2.template zeros_like<T>();
        // Matrix zeros2_like_input = mat2.template zeros_like<T2>();
    }
};

int main()
{
    return 0;
}