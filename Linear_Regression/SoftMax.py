import numpy as np
from Base_Algorithms.Linear_Regression import Fashion_Mnist as fm


# 当标签为离散值时，前面的线性回归不再适用，可以改用softmax
# 输入是一个m*n的矩阵, m为样本数量, n为特征数, 输出是每个样本对应的标签的概率
# 在这里假设标签用的是各个类别的索引,从0到q-1,q为类别的数量
# 参数w, b
# 这次就不写sgd了


class SoftMax:
    def __init__(self, X, y, class_num, learning_rate, iteration, class_name=None):
        self._output_num = len(y)
        self._class_num = class_num
        self._data_row = X.shape[0]
        self._data_line = X.shape[1]
        self._X = X
        self._output = np.zeros((self._data_row, self._class_num))
        self._iteration = iteration
        self._learning_rate = learning_rate
        temp_matrix = np.zeros((self._X.shape[0], self._class_num))
        # one-hot编码
        for i, data in zip(range(len(y)), y):
            temp_matrix[i][data] = 1
        self._y = temp_matrix
        self._label = y
        self._class_name = class_name
        self._w = np.zeros((self._X.shape[1], self._class_num))
        self._b = np.zeros(self._class_num).reshape(1, self._class_num)

    # 给出每个用例对应输出类型的概率，计算最大的那一个
    def forward(self):
        output = np.dot(self._X, self._w) + self._b
        output_sum = np.sum(np.exp(output), axis=1) \
            .reshape(self._X.shape[0], 1)
        self._output = np.exp(output) / output_sum
        return self._output

    def loss(self):
        output = self._output
        pred = self._y * output
        pred = np.sum(pred, axis=1)
        loss = np.sum(-np.log(pred + 1e-5)) / len(output)
        return loss

    def accuracy(self):
        max_index = self.predict()
        return np.sum(max_index == self._label) / len(self._label) * 100

    # 更新权重
    def backwards(self):
        # 关键梯度
        temp = self._output - self._y
        grad_w = np.dot(self._X.T, temp) / self._X.shape[0]
        grad_b = np.sum(temp, axis=0) / self._X.shape[0]
        grad_b = grad_b.reshape(1, self._class_num)
        self._w -= self._learning_rate * grad_w
        self._b -= self._learning_rate * grad_b

    def predict(self):
        output = self._output
        max_index = np.argmax(output, axis=1)
        return max_index

    def predict_class(self):
        max_index = self.predict()
        if len(self._class_name) == self._output_num:
            class_list = []
            for index in max_index:
                class_list.append(self._class_name[index])
            return class_list
        else:
            print('类别列表输入有误,可以调用predict查看索引')

    def train(self):
        for i in range(self._iteration):
            self.forward()
            loss = self.loss()
            acc = self.accuracy()
            print(f'iteration {i + 1}: Loss: {loss}   accuracy: {acc}')
            self.backwards()


if __name__ == '__main__':
    train_data, train_label, class_names = fm.get_data()
    s = SoftMax(train_data, train_label, 10, 0.2, 1000, class_names)
    s.train()
