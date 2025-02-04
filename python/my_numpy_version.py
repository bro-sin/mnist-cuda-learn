#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from dataclasses import dataclass


# In[2]:


batch_size=4


# In[3]:


data_dir = "data"

mnist_data_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081)),
    ]
)

train_dataset = datasets.MNIST(
    root=data_dir, train=True, transform=mnist_data_transform, download=True
)
test_dataset = datasets.MNIST(
    root=data_dir, train=False, transform=mnist_data_transform, download=True
)


# In[4]:


train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


# In[5]:


class Linear:
    def __init__(
        self, input_features: int, output_features: int, with_bias: bool = True
    ):
        self.weights = Linear.initialize_weights(
            input_size=input_features, output_size=output_features
        )
        self.grad_weights = np.empty_like(self.weights)
        self.grad_input = np.empty((batch_size,input_features))
        if with_bias:
            self.bias = Linear.initialize_bias(output_size=output_features)
            self.grad_bias = np.empty_like(self.bias)

    def zero_grad(self):
        self.grad_weights.fill(0)
        self.grad_bias.fill(0)
        self.grad_input.fill(0)

    def forward(self, x: np.ndarray):
        assert x.shape[1] == self.weights.shape[0]
        return np.dot(x, self.weights) + self.bias

    def initialize_weights(input_size, output_size):
        return np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)

    def initialize_bias(output_size):
        return np.zeros((1, output_size))

    def __call__(self, x: np.ndarray):
        return self.forward(x)

    def backward(
        self,
        grad_output: np.ndarray,
        x: np.ndarray,
    ):
        # 计算的是这个线性层有变化，然后对线性层相关变量的梯度
        grad_weights = x.T @ grad_output
        # TODO: 搞清楚grad_output的维度
        grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = grad_output @ self.weights.T
        # breakpoint()
        self.grad_weights[:] = grad_weights[:]
        self.grad_bias[:] = grad_bias[:]
        self.grad_input[:] = grad_input[:]
        return grad_input, grad_weights, grad_bias


# In[6]:


class ReLU:
    def __init__(self):
        pass

    def forward(self, input: np.ndarray):
        return np.maximum(0, input)

    def backward(self, grad_output: np.ndarray, input: np.ndarray):
        return grad_output * (input > 0).astype(float)

    def __call__(self, x: np.ndarray):
        return self.forward(x)


# In[7]:


class Softmax:
    def __init__(self):
        pass

    def __call__(self, input: np.ndarray):
        return self.forward(input=input)

    def forward(self, input: np.ndarray):
        exp_input = np.exp(input - np.max(input, axis=1, keepdims=True))
        return exp_input / np.sum(exp_input, axis=1, keepdims=True)

    def backward(self, grad_output: np.ndarray, input: np.ndarray):
        raise NotImplementedError


# In[8]:


class CrossEntropyLoss:
    def __init__(self):
        self.softmax: Softmax = Softmax()
        pass

    def forward(self, input: np.ndarray, target: np.ndarray):
        """
        类似pytorch中的版本
        input: (N, C)，在这个jupyter notebook中，C=10，N应该会等于4,这里是没有归一化的神经网络上一层输出
        target: (N, ),也就是预测的标签,预测的标签这个输入形式意味着其等价的（N,C)这种维度的向量表示中，每一个样本只能对应一个种类，也就是只有一个1,其余的全是0
        返回一个非负数，当且仅当input中预测完全正确的时候才等于0
        """
        # 对input归一化
        input_probs = self.softmax(input=input)
        # 找到每一个样本中正确的那一个分类的概率（因为错误的分类的真值是0,所以最终不会被累加）
        assert len(input_probs) == len(target)
        correct_class_probs = input_probs[np.arange(len(input_probs)), target]
        # 这意味着如下代码：
        # correct_class_probs = np.zeros_like(target)
        # for i, input_prob, target_label in enumerate(zip(input_probs, target)):
        #     #将正确结果那一个的预测分类概率记下来
        #     correct_class_probs[i] = input_prob[target_label]

        # 返回交叉熵损失
        return -np.mean(np.log(correct_class_probs))
    def __call__(self, input: np.ndarray, target: np.ndarray):
        return self.forward(input=input,target=target)


# In[9]:


@dataclass
class Optimizer:
    learning_rate: float
    linear_weight1: np.ndarray
    linear_bias1: np.ndarray
    linear_weight2: np.ndarray
    linear_bias2: np.ndarray
    linear_grid_weight1: np.ndarray
    linear_grid_bias1: np.ndarray
    linear_grid_weight2: np.ndarray
    linear_grid_bias2: np.ndarray

    def update_weights(self):
        self.linear_weight1[:] -= self.learning_rate * self.linear_grid_weight1
        self.linear_bias1[:] -= self.learning_rate * self.linear_grid_bias1
        self.linear_grid_weight2[:] -= self.learning_rate * self.linear_grid_weight2
        self.linear_grid_bias2[:] -= self.learning_rate * self.linear_grid_bias2


# In[10]:


class MLP:
    def __init__(self, input_features: int, hidden_features: int, num_classes: int):
        self.fc1: Linear = Linear(
            input_features=input_features, output_features=hidden_features
        )
        self.relu: ReLU = ReLU()
        self.fc2: Linear = Linear(
            input_features=hidden_features, output_features=num_classes
        )
        self.softmax: Softmax = Softmax()

    def __call__(self, x: np.ndarray):
        return self.forward(x=x)

    def forward(self, x: np.ndarray):
        fc1_input = x.reshape(batch_size, 28 * 28)
        fc1_output = self.fc1(fc1_input)
        relu_output = self.relu(fc1_output)
        fc2_output = self.fc2(relu_output)
        return fc2_output, (fc1_input, fc1_output, relu_output)

    def backward(self, grad_output, cache):
        fc1_input, fc1_output, relu_output = cache
        # 计算线性层fc2的梯度
        grad_fc2, grad_weights2, grad_bias2 = self.fc2.backward(
            grad_output=grad_output,
            x=relu_output,  # 这里与forward的时候的输入relu_output要对应
        )
        # 计算ReLU的梯度
        grad_relu = self.relu.backward(
            grad_output=grad_fc2,
            input=fc1_output,
        )
        # 计算线性层fc1的梯度
        grad_fc1, grad_weights1, grad_bias1 = self.fc1.backward(
            grad_output=grad_relu,
            x=fc1_input,
        )
        return grad_weights1, grad_bias1, grad_weights2, grad_bias2


# In[11]:


def train(model: MLP, train_loader: DataLoader, learning_rate: float, epochs: int):
    optimizer = Optimizer(
        learning_rate=learning_rate,
        linear_weight1=model.fc1.weights,
        linear_bias1=model.fc1.bias,
        linear_weight2=model.fc2.weights,
        linear_bias2=model.fc2.bias,
        linear_grid_weight1=model.fc1.grad_weights,
        linear_grid_bias1=model.fc1.grad_bias,
        linear_grid_weight2=model.fc2.grad_weights,
        linear_grid_bias2=model.fc2.grad_bias,
    )
    criterion = CrossEntropyLoss()

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for i, (images, labels) in enumerate(train_loader):
            y_pred, cache = model(x=images)
            #计算loss
            ce_loss=criterion(input=y_pred,target=labels)
            print(f"loss = {ce_loss}")
            softmax_probs = model.softmax(input=y_pred)
            y_true_one_hot = np.zeros_like(y_pred)
            y_true_one_hot[range(batch_size), labels] = 1

            grad_output = softmax_probs - y_true_one_hot
            model.backward(grad_output=grad_output, cache=cache)
            optimizer.update_weights()
        


# In[12]:


input_size = 28 * 28
hidden_size = 256
output_size = 10
model = MLP(
    input_features=input_size, hidden_features=hidden_size, num_classes=output_size
)
epochs=3
learning_rate=1e-3
train(model=model,train_loader=train_loader,learning_rate=learning_rate,epochs=epochs)

