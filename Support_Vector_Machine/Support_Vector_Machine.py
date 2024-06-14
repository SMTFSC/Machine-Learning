import random

import numpy as np
import matplotlib.pyplot as plt


# 主要理论来源: <<统计学习方法>> 李航
# SMO算法
# 主要解决问题:二次凸优化问题
# 由于二次凸优化问题的拉格朗日乘数法在数据量很大时计算很复杂(涉及到高斯消元/奇异值分解),
# 可以类比线性回归中的解析解和数值解
# 因此引入序列最小化算法(SMO)
# 主要思想是选取两个a进行优化,两个a分别从外循环和内循环中选取,可以极大加速问题的求解

# kkt条件:
# 1. a = 0 <=> y*g(x) >= 1
# 2. 0 < a < c <=> y*g(x) = 1
# 3. a = c <=> y*g(x) <= 1
# 4. g(x) = E(x) + y

# 由第四条可以转换一下kkt条件
# kkt条件(转换后)
# 1. a = 0 <=> y*E(x) >= 0
# 2. 0 < a < c <=> y*E(x) = 0
# 3. a = c <=> y*E(x) <= 0

# 如果是软间隔，还可以加上容忍度tor
# kkt条件(软间隔)
# 1. a = 0 <=> y*E(x) >= -tor
# 2. 0 < a < c <=> y*E(x) = 0
# 3. a = c <=> y*E(x) <= tor

# 其中第一个a选择的思路是:
# 首先遍历全部的0 < a < C(为什么优先选择这些点，因为他们的条件最严格，同时它们也最有可能违反kkt条件),
# 检查它们是否满足kkt条件，如果满足，那么就遍历所有a
# 检查他们是否满足kkt条件
# 如果都满足，那么遍历结束，如果找到了，那么重新回到第一步
# 然后是找第二个点的思路:
# 找到第一个点后遍历所有a, 找到那个使|Ej-Ei|最大的那个点进行优化,这个地方也是在加速优化


# note: 已完成无核函数版本
# 未完成:
# 核函数来分类线性不可分数据
# 目前只支持二分类，如何拓展到多分类?
class Support_Vector_Machine:
    def __init__(self, X, y, max_iteration, C, tolerance, kernel=None, sigma=1):
        """

        :param X: 样本集
        :param y: 标签
        :param max_iteration: 最大迭代次数
        :param C: 惩罚参数
        :param tolerance: 容忍度
        :param kernel: 默认不引入kernel, 如果要引入,可以指定参数
        1.kernel = 'Gaussian' 高斯核函数
        2.kernel = 'polynomial' 多项式核函数
        """

        self._X = X
        self._y = y
        self._X_row = self._X.shape[0]
        self._X_line = self._X.shape[1]
        # 初始化a
        self._a = np.zeros(self._X_row)
        # 初始化b
        self._b = 0
        self._max_iteration = max_iteration
        # 惩罚系数
        self._C = C
        # 容忍程度
        self._tolerance = tolerance
        # 保存E值
        self._E_Cache = np.zeros((self._X_row, 2))
        # 记录kernel的值
        self._kernel = kernel
        # 记录sigma值
        self._sigma = sigma

    def train(self):
        iteration = 0
        # 这个变量用来说明是否遍历全部a
        whether_to_traverse_all = True
        # 这个变量用来标记有多少a被选取
        num_a_is_selected = 0
        while (iteration < self._max_iteration) and (whether_to_traverse_all or (num_a_is_selected > 0)):
            num_a_is_selected = 0
            if whether_to_traverse_all:
                # 遍历全部a
                for i in range(self._X_row):
                    num_a_is_selected += self.inner_loop(i)
                iteration += 1
            else:
                # 如果不遍历全部，那么就选取最有可能违反kkt条件的值,即 0 < a < C的那些值
                indexes = np.intersect1d(np.where(self._a < self._C),
                                         np.where(self._a > 0))
                for index in indexes:
                    num_a_is_selected += self.inner_loop(index)
                iteration += 1
            # 这个地方的条件决定了是否停止while
            # 如果遍历全部a也没有找到num_a_is_selected > 0，那么说明没有a可选，结束循环
            # 如果遍历全部a找到了num_a_is_selected > 0, 那么有a 改变了，可以尝试找一下违反kkt的a
            if whether_to_traverse_all:
                whether_to_traverse_all = False
            elif num_a_is_selected == 0:
                whether_to_traverse_all = True

    # 注意，当使用kernel时，要把所有np.dot换成kernel
    # 这里就不再转换了
    # 为了与标签区分，这里使用大写的Y
    # 先不写核函数的SVM
    def kernel(self, X, Y, sigma):
        if len(Y.shape) == 1:
            Y = Y.reshape(1, Y.shape[0])
            column = 1
        else:
            column = Y.shape[0]
        row = X.shape[0]
        if self._kernel == 'Gaussian':
            result = np.zeros((row, column))
            for i in range(column):
                result[:, i] = np.exp(-np.sum((X - Y[i].reshape(1, Y.shape[1])) ** 2, axis=1) / 2 * sigma * sigma)
            return result

    def inner_loop(self, i):
        Ei = self.calculate_E(i)
        # 这个地方说明违反了kkt条件
        if (((self._y[i] * Ei) < -self._tolerance and self._a[i] < self._C) or
                ((self._y[i] * Ei) > self._tolerance and self._a[i] > self._C)):
            j, Ej = self.select_j(i, Ei)
            a_old_i = self._a[i].copy()
            a_old_j = self._a[j].copy()
            if self._y[i] != self._y[j]:
                L = max(0, self._a[j] - self._a[i])
                H = min(self._C, self._C + self._a[j] - self._a[i])
            else:
                L = max(0, self._a[j] + self._a[i] - self._C)
                H = min(self._C, self._a[j] + self._a[i])
            # 如果最小值等于最大值，说明aj不会改变
            if L == H:
                return 0
            if not self._kernel:
                eta = np.dot(self._X[i], self._X[i]) + np.dot(self._X[j], self._X[j]) - \
                      2 * np.dot(self._X[i], self._X[j])
                # 这个地方当eta <= 0是不能直接用a的更新公式，要改用更高级的
                # 二阶优化策略，这个地方先不管
                if eta <= 0:
                    return 0
                # 更新aj
                self._a[j] += self._y[j] * (Ei - Ej) / eta
                # 加上裁剪a
                self._a[j] = self.clip_a(L, H, self._a[j])
                # 如果步长不够,也不更新
                if abs(self._a[j] - a_old_j) < 1e-5:
                    return 0
                # 更新ai
                self._a[i] += self._y[i] * self._y[j] * (a_old_j - self._a[j])

                b_new_1 = self._b - Ei - self._y[i] * (self._a[i] - a_old_i) * np.dot(self._X[i], self._X[i]) - self._y[
                    j] * (self._a[j] - a_old_j) * np.dot(self._X[i], self._X[j])
                b_new_2 = self._b - Ej - self._y[j] * (self._a[j] - a_old_j) * np.dot(self._X[j], self._X[j]) - self._y[
                    i] * (self._a[i] - a_old_i) * np.dot(self._X[i], self._X[j])
                # 更新参数b
                if 0 < self._a[i] < self._C:
                    self._b = b_new_1
                elif 0 < self._a[j] < self._C:
                    self._b = b_new_2
                else:
                    self._b = (b_new_2 + b_new_1) / 2
                # 更新E的值要放在b的后面
                # 更新Ej
                print('update EJ')
                self.update_E(j)
                # 更新Ei
                self.update_E(i)

                return 1
        else:
            return 0

    def calculate_w(self):
        w = np.sum((self._a * self._y).reshape(self._X_row, 1) * self._X, axis=0)
        return w

    def plot_2d(self):
        if self._X_line == 2:
            # draw the graph
            plt.scatter(self._X[self._y == -1][:, 0], self._X[self._y == -1][:, 1], color='red', label='neg')
            plt.scatter(self._X[self._y == 1][:, 0], self._X[self._y == 1][:, 1], color='blue', label='pos')
            plt.legend()
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title('Scatter plot of two-feature dataset')

            x_min, x_max = self._X[:, 0].min() - 1, self._X[:, 0].max() + 1

            # 计算 y 的值
            x_values = np.linspace(x_min, x_max, 5)
            w = self.calculate_w()
            y_values = -(w[0] * x_values + self._b) / w[1]

            # 绘制直线
            plt.plot(x_values, y_values, color='green', label='Decision Boundary')
            plt.show()

    def select_j(self, i, Ei):
        # 寻找j
        valid_indexes = np.where(self._E_Cache[:, 0] != 0)[0]
        result_index = -1
        max_delta = 0
        result_E = 0
        if len(valid_indexes) > 1:
            for index in valid_indexes:
                if index == i:
                    continue
                E_index = self.calculate_E(index)
                if abs(E_index - Ei) > max_delta:
                    max_delta = abs(self.calculate_E(index) - Ei)
                    result_index = index
                    result_E = E_index
            return result_index, result_E
        else:
            # 第一次时,E全为0，可以随机选一个aj
            j = i
            while j == i:
                j = int(random.uniform(0, self._X_row))
            Ej = self.calculate_E(j)
            return j, Ej

    # 计算E值
    def calculate_E(self, index):
        X_i = self._X[index]
        E_i = np.sum(self._a * self._y * np.dot(self._X, X_i)) + \
              self._b - self._y[index]
        return E_i

    def clip_a(self, L, H, a):
        if a > H:
            return H
        elif a < L:
            return L
        else:
            return a

    def update_E(self, index):
        self._E_Cache[index] = [1, self.calculate_E(index)]

    def predict(self, index):
        w = self.calculate_w()
        value = np.dot(self._X[index], w) + self._b
        print(f'predict: {value}')
        print(f'label: {self._y[index]}')

    def accuracy(self):
        w = self.calculate_w()
        predict = np.dot(self._X, w) + self._b
        predict = np.where(predict > 0, 1, -1)
        acc = np.sum(predict == self._y) / self._X_row * 100
        return acc


if __name__ == '__main__':
    with open('testSet.txt') as f:
        lines = f.readlines()
        train_data = []
        train_label = []
        for line in lines:
            array = line.strip().split()
            train_data.append(list(map(float, array[:2])))
            train_label.append(int(array[2]))
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    # 调整惩罚参数的值可以改变超平面的位置
    s = Support_Vector_Machine(train_data, train_label, 100, 15, 0.001, 'Gaussian')
    #    s.train()
    s.predict(2)
    # 事实上这个准确率是100
    print(s.accuracy())
#    s.plot_2d()
