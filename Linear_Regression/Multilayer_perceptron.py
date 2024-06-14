# 线性回归和softmax回归都是单层,mlp引入隐藏层
import numpy as np
from Base_Algorithms.Linear_Regression import Fashion_Mnist as fm


# 先构建一个两层感知机
# 输入层为 m*d
# 输出层为 m*n
# 隐藏层大小为 h 作为超参数 可调
# 隐藏层的输入参数为 d*h
# 隐藏层的输出参数为 h*n
# 激活函数选用relu  当然也可以选用sigmoid tanh
# 激活函数的目的主要是为了引入非线性
# 仍然使用fashion-mnist数据集进行分类
# 用softmax函数来预测概率,用cross entropy来计算损失

class Multilayer_Perceptron:
    def __init__(self, X, y, hidden_num, class_num, learning_rate, iteration):
        self._X = X
        self._train_data_row = self._X.shape[0]
        self._train_data_line = self._X.shape[1]
        self._class_num = class_num
        self._label = y
        matrix = np.zeros((self._train_data_row, class_num))
        # one-hot 编码
        for i, data in zip(range(len(y)), y):
            matrix[i][data] = 1
        self._y = matrix
        self._hidden_num = hidden_num
        self._learning_rate = learning_rate
        self._iteration = iteration
        # 初始化时不能全部初始化为0，不然梯度无法传播
        # 但也不能直接使用np.random.random,要对数据进行缩放
        self._w_input_to_hidden = np.random.random((self._train_data_line, self._hidden_num)) * 0.001
        self._b_input_to_hidden = np.random.random((1, self._hidden_num)) * 0.001
        self._w_hidden_to_output = np.random.random((self._hidden_num, self._class_num)) * 0.001
        self._b_hidden_to_output = np.random.random((1, self._class_num)) * 0.001
        self._output_from_input_to_hidden = np.zeros((self._train_data_row, self._hidden_num))
        self._output_from_hidden_to_output = np.zeros((self._train_data_row, self._class_num))
        self._output_with_softmax = np.zeros((self._train_data_row, self._class_num))
        self._relu_gradient = np.zeros((self._train_data_row, self._hidden_num))
        self._relu_output_from_input_to_hidden = np.zeros((self._train_data_row, self._hidden_num))

    def forward(self):
        self._output_from_input_to_hidden = np.dot(self._X, self._w_input_to_hidden) + \
                                            self._b_input_to_hidden
        # 这个地方要relu
        self.relu()
        self._output_from_hidden_to_output = np.dot(self._relu_output_from_input_to_hidden,
                                                    self._w_hidden_to_output) + self._b_hidden_to_output
        self.softmax()

    def loss(self):
        pred = self._y * self._output_with_softmax
        pred = np.sum(pred, axis=1)
        loss = np.sum(-np.log(pred)) / self._train_data_row
        return loss

    def backwards(self):
        temp = self._output_with_softmax - self._y
        grad_w_hidden_to_output = np.dot(self._relu_output_from_input_to_hidden.T, temp) / self._train_data_row
        grad_b_hidden_to_output = np.sum(temp, axis=0) / self._train_data_row
        grad_b_hidden_to_output = grad_b_hidden_to_output.reshape((1, self._class_num))
        temp2 = np.dot(temp, self._w_hidden_to_output.T) * self._relu_gradient
        grad_w_input_to_hidden = np.dot(self._X.T, temp2) / self._train_data_row
        grad_b_input_to_hidden = np.sum(temp2, axis=0) / self._train_data_row
        grad_b_input_to_hidden = grad_b_input_to_hidden.reshape((1, self._hidden_num))
        self._w_hidden_to_output -= self._learning_rate * grad_w_hidden_to_output
        self._b_hidden_to_output -= self._learning_rate * grad_b_hidden_to_output
        self._w_input_to_hidden -= self._learning_rate * grad_w_input_to_hidden
        self._b_input_to_hidden -= self._learning_rate * grad_b_input_to_hidden

    def train(self):
        for i in range(self._iteration):
            self.forward()
            loss = self.loss()
            accuracy = self.accuracy()
            print(f'iteration: {i + 1}  loss: {loss}  accuracy: {accuracy}')
            self.backwards()

    def accuracy(self):
        predict_class = np.argmax(self._output_with_softmax, axis=1)
        return np.sum(predict_class == self._label) / self._train_data_row * 100

    def softmax(self):
        temp_sum = np.sum(np.exp(self._output_from_hidden_to_output), axis=1).reshape(self._train_data_row, 1)
        self._output_with_softmax = np.exp(self._output_from_hidden_to_output) / temp_sum

    def relu(self):
        self._relu_gradient = \
            np.where(self._output_from_input_to_hidden > 0, 1, 0)
        # relu
        self._relu_output_from_input_to_hidden = \
            np.maximum(self._output_from_input_to_hidden, 0)


if __name__ == '__main__':
    train_data, train_label, class_names = fm.get_data()
    # 超参数还可以调一下
    m = Multilayer_Perceptron(train_data, train_label, 256, 10, 0.5, 1000)
    m.train()
