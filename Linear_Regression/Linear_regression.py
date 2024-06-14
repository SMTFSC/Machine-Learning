import numpy as np


class Linear_Regression:
    """
    线性回归:

    包含两部分:

    1.解析解，利用矩阵

    2.数值解，利用梯度下降，包含三种梯度下降:
        1.批量梯度下降

        2.随机梯度下降

        3.小批量梯度下降
    """
    def __init__(self, X, y, iteration, learning_rate, kind='numeric_solution', gradient_kind='BGD', batch=0):
        self._X = X
        self._y = y
        self._w = np.zeros(self._X.shape[1])
        self._b = 0
        self._iteration = iteration
        self._learning_rate = learning_rate
        self._kind = kind
        self._gradient_kind = gradient_kind
        self._batch = batch
        np.random.seed(0)

    def get_loss(self, y_predict):
        return np.sum((self._y - y_predict) ** 2 / 2) / len(self._y)

    def forward(self):
        return np.dot(self._X, self._w) + self._b

    def get_gradient(self, y_pred):
        """
        梯度下降算法的缺点:

        1.局部最小点和鞍点

        2.梯度消失和梯度爆炸

        当gradient_kind == 'BGD' 时是使用所有数据

        当gradient_kind == 'SGD' 时是使用一行数据

        当gradient_kind == 'MBGD' 时是使用一部分数据
        """
        if self._gradient_kind == 'BGD':
            return np.dot(self._X.T, y_pred - self._y) / self._X.shape[0], \
                   np.sum(y_pred - self._y) / self._X.shape[0]
        elif self._gradient_kind == 'SGD':
            random_index = np.random.randint(0, self._X.shape[0] - 1)
            data = self._X[random_index].T
            return data * (y_pred[random_index] - self._y[random_index]), \
                y_pred[random_index] - self._y[random_index]
        elif self._gradient_kind == 'MBGD':
            if self._batch == 0 or self._batch > self._X.shape[0]:
                print('批次不合法,要进行MSGD,请检查输入的batch')
            else:
                random_list = np.random.randint(0, self._X.shape[0]-1, (self._batch,))
                data = self._X[random_list]
                return np.dot(data.T,  y_pred[random_list] - self._y[random_list]) / len(data), \
                    np.sum(y_pred[random_list]-self._y[random_list]) / len(data)
        else:
            print('输入梯度类型不存在')

    def backwards(self, y_pred):
        grad_w, grad_b = self.get_gradient(y_pred)
        self._w -= grad_w * self._learning_rate
        self._b -= grad_b * self._learning_rate

    def train(self):
        if self._kind != 'analytical_solution' and self._kind != 'numeric_solution':
            print('无指定类型')
            return
        if self._kind == 'analytical_solution':
            try:
                temp_x = np.linalg.inv(np.dot(self._X.T, self._X))
                self._w = np.dot(np.dot(temp_x, self._X.T), self._y)
                return self._w
            except Exception:
                print('无法求解析解，将自动开始进行数值解')
        for i in range(self._iteration):
            y_pred = self.forward()
            loss = self.get_loss(y_pred)
            print(f'iteration {i + 1} : {loss}')
            self.backwards(y_pred)
        return self._w


m_train = 1000
n_train = 1000
# 根据解析解和数值解的差值可以看出,当迭代次数加深,数值解在不断逼近解析解
if __name__ == '__main__':
    X = np.random.random((m_train, n_train))
    y = np.random.random(m_train)
    linear_regression = Linear_Regression(X, y, 2000, 0.005)
    result1 = linear_regression.train()
    l = Linear_Regression(X, y, 1000, 0.005, kind='analytical_solution')
    result2 = l.train()
    # print(f'数值解和解析解的误差: {np.sum(abs(result1-result2))/X.shape[0]}')
    # 可以看出解析解的误差几乎为0, 所以可以将解析解的权重作为基准
    print(f'数值解的误差: {linear_regression.get_loss(linear_regression.forward())}')
    # print(f'解析解的误差: {l.get_loss(l.predict())}')
