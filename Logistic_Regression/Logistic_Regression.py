import numpy as np


class Logistic_Regression:
    def __init__(self, X, y, learning_rate, iteration):
        self._X = X
        self._y = y
        self._train_data_row = self._X.shape[0]
        self._train_data_line = self._X.shape[1]
        self._y_len = len(self._y)
        self._iteration = iteration
        self._learning_rate = learning_rate
        self._w = np.zeros(self._train_data_line)
        self._b = 0
        self._predict = None

    def train(self):
        for i in range(self._iteration):
            self.forward()
            loss = self.loss()
            accuracy = self.accuracy()
            print(f'iteration {i + 1}  loss:{loss}  accuracy:{accuracy}')
            self.backwards()

    # 注意要经过sigmoid
    def forward(self):
        self._predict = self.sigmoid(np.dot(self._X, self._w) + self._b)
        return self._predict

    def backwards(self):
        # 改用随机梯度下降试一下,随机梯度太过于随机，每次结果不一样，很难评估
        temp = self._predict - self._y
        grad_w = np.dot(self._X.T, temp) / self._train_data_row
        grad_b = temp / self._train_data_row
        self._w -= self._learning_rate * grad_w
        self._b -= self._learning_rate * grad_b
        # 把学习率降低，防止出现来回振荡现象
        self._learning_rate -= self._learning_rate / 50

    def predict(self, new_data):
        possibility = self.forward()
        if possibility >= 0.5:
            print('1')
        else:
            print('0')

    # 采用交叉熵损失来衡量
    def loss(self):
        return np.sum(-(self._y * np.log(self._predict) + (1 - self._y) *
                        np.log(1 - self._predict))) / self._train_data_row

    def accuracy(self):
        result = np.where(self._predict >= 0.5, 1, 0)
        return np.sum(result == self._y) / self._y_len * 100

    def sigmoid(self, predict):
        return 1 / (1 + np.exp(-predict))


def read():
    with open('horseColicTraining.txt') as f:
        train_data = []
        train_label = []
        lines = f.readlines()
        for line in lines:
            data = line.strip().split()
            train_data.append(list(map(float, data[:-1])))
            train_label.append(int(float(data[-1])))
        train_data = np.array(train_data)
        train_label = np.array(train_label)
        return train_data, train_label


if __name__ == '__main__':
    train_data, train_label = read()
    # 准确率只能达到72,貌似陷入了局部最优,不过可能是数据本身的问题
    # 总体来说还是可以的
    l = Logistic_Regression(train_data, train_label, 0.05, 500)
    l.train()
