import numpy as np


def show_image(image):
    for i in range(28):
        for j in range(28):
            pixel = image[i * 28 + j]
            if pixel > 0:
                print("x", end="")
            else:
                print(" ", end="")
        print()


input_size = 28 * 28
hidden_size = 256
output_size = 10
batch_size = 4
# 加载训练数据
train_dataset = np.fromfile("./mnist_data/X_train.bin", dtype=np.float32).reshape(
    input_size, -1, order="F"
)
print(train_dataset.shape)

start_train_x = train_dataset[:, :batch_size]
show_image(start_train_x[:, 0])
show_image(start_train_x[:, 1])


# 展示按行求和结果
print(start_train_x.sum(axis=0))


# 加载权重
fc1_weights: np.ndarray = np.fromfile(
    "./python/numpy_init_weights/fc1.bin", dtype=np.float32
).reshape(hidden_size, input_size)
print(fc1_weights.shape)
# print(fc1_weights)

# 计算乘法结果
fc1_output = fc1_weights.dot(start_train_x)
print(fc1_output.shape)
# print(fc1_output)
