import numpy as np

def test_multiply():
    # 测试行优先矩阵乘法
    a = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])
    b = np.array([
        [7.0, 8.0],
        [9.0, 10.0],
        [11.0, 12.0]
    ])
    expected = np.array([
        [58.0, 64.0],
        [139.0, 154.0]
    ])
    result = np.dot(a, b)
    assert np.array_equal(result, expected), "Row major multiplication failed"
    print("Row major multiplication passed")

    # 测试列优先矩阵乘法
    a = np.array([
        [1.0, 4.0],
        [2.0, 5.0],
        [3.0, 6.0]
    ]).T
    b = np.array([
        [7.0, 9.0, 11.0],
        [8.0, 10.0, 12.0]
    ]).T
    expected = np.array([
        [58.0, 139.0],
        [64.0, 154.0]
    ]).T
    result = np.dot(a, b)
    assert np.array_equal(result, expected), "Column major multiplication failed"
    print("Column major multiplication passed")

    # 测试行优先和列优先混合矩阵乘法
    a = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])
    b = np.array([
        [7.0, 9.0, 11.0],
        [8.0, 10.0, 12.0]
    ]).T
    expected = np.array([
        [58.0, 64.0],
        [139.0, 154.0]
    ])
    result = np.dot(a, b)
    assert np.array_equal(result, expected), "Mixed major multiplication failed"
    print("Mixed major multiplication passed")

    # 测试不兼容维度的矩阵乘法
    a = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])
    b = np.array([
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0]
    ])
    try:
        result = np.dot(a, b)
        assert False, "Incompatible dimensions multiplication should have failed"
    except ValueError:
        print("Incompatible dimensions multiplication passed")

if __name__ == "__main__":
    test_multiply()