import numpy as np
import matplotlib.pyplot as plt


# K-means算法较为简单,就不详细叙述了
# 但是K-means算法有个很大的缺陷,就是容易陷入局部最优
class K_Means:
    def __init__(self, X, num):
        self._X = X
        self._num = num
        self._pivot = np.random.random((self._num, 2))

    def plot(self):
        plt.scatter(self._X[:, 0], self._X[:, 1], color='blue', label='data')
        plt.xlabel('f1')
        plt.ylabel('f2')
        plt.legend()
        plt.scatter(self._pivot[:, 0], self._pivot[:, 1], color='red', label='pivot')
        plt.show()

    def train(self):
        for i in range(200):
            temp_pivot = np.zeros((self._num, 2))
            temp_pivot_num = np.zeros(self._num)
            for data in self._X:
                index = np.argmin(np.sum((self._pivot - data) ** 2, axis=1))
                temp_pivot[index] += data
                temp_pivot_num[index] += 1
            # 重新计算pivot
            temp_pivot = temp_pivot / temp_pivot_num.reshape(self._num, 1)
            if np.any(temp_pivot == self._pivot):
                break
            self._pivot = temp_pivot


if __name__ == '__main__':
    with open('testSet2.txt') as f:
        train_data = []
        lines = f.readlines()
        for line in lines:
            train_data.append(list(map(float, line.split())))
        train_data = np.array(train_data)
    k = K_Means(train_data, 3)
    k.train()
    k.plot()
