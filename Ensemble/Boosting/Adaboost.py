import numpy as np
from Base_Algorithms.Logistic_Regression.Logistic_Regression import read


# Adaboost means adaptive boosting
# 核心思想是用多个弱分类器集成变成一个强分类器
# 每个弱分类器有一个权重 a
# a 的计算公式为 a = 1/2 * ln(1-theta / theta )
# theta = 误分的样本数 / 总的样本数
# 样本权重更新公式为
# 正确分类的样本:
# D = D * pow(e, -a) / sum(D)
# 误分类样本:
# D = D * pow(e, a) / sum(D)


# 步骤：
# 1.找到一个单层决策树(当然也可以是其他的弱分类器)使误分权重最小
# 2.然后返回预测的标签
# 3.然后更新该分类器的权重
# 4.然后更新每个样本的权重,重复第一步

# 总结,该模型缺少对新数据的预测功能(没有保存每个弱分类器的相应参数,需要改进一下(先不改了))

# 数据集依然使用病马数据集(看一下logistic regression 和 boosting 的区别)
class Adaboost:
    def __init__(self, X, y, iteration):
        self._X = X
        self._y = y
        print(self._y)
        self._X_row = self._X.shape[0]
        self._X_column = self._X.shape[1]
        self._iteration = iteration
        self._Weight = np.ones(self._X_row) / self._X_row
        # 保存每个弱分类器的权重
        self._a = np.zeros(self._iteration)
        # 保存每个分类器的预测结果
        self._pred_matrix = np.zeros((self._iteration, self._X_row))

    def train(self):
        for i in range(self._iteration):
            best_wrong_sum, best_pred, best_final_pred = self.best_stump()
            self._pred_matrix[i] = best_final_pred
            theta = best_wrong_sum

            a = 1 / 2 * np.log((1 - theta) / theta)
            self._a[i] = a

            # 正确的样本
            correct_indexes = np.array([int(best_pred[j] == self._y[j]) for j in range(self._X_row)])
            Weight_correct = self._Weight * correct_indexes * np.exp(-a)
            # 错误的样本
            wrong_indexes = np.array([int(best_pred[j] != self._y[j]) for j in range(self._X_row)])
            Weight_wrong = self._Weight * wrong_indexes * np.exp(a)
            # 更新权重,要保证权重之和为1
            self._Weight = (Weight_correct + Weight_wrong) / np.sum(self._Weight)
            accuracy = self.accuracy()
            print(f'{i + 1}个弱学习器, accuracy: {accuracy}')

    def accuracy(self):
        final_predict = np.sum(self._pred_matrix * self._a.reshape(self._iteration, 1), axis=0)
        final_predict = np.where(final_predict > 0, 1, 0)
        accuracy = np.sum(final_predict == self._y) / self._X_row * 100
        return accuracy

    # 生成弱分类器
    def best_stump(self):
        best_wrong_sum = np.inf
        best_pred = None
        for i in range(self._X_column):
            min_value = min(self._X[:, i])
            max_value = max(self._X[:, i])
            step = (max_value - min_value) / 10
            for j in range(-1, 11):
                threshold = min_value + j * step
                # 注意,这个地方可以双向,不能只考虑单单向
                # 单个决策树也要考虑全面,不然只能得到60多的准确率(血的教训)
                for d in ['lt', 'gt']:
                    if d == 'lt':
                        pred = np.where(self._X[:, i] <= threshold, 0, 1)
                        # 有误, 因为每个样本的权值不一样
                        wrong_sum = np.sum(
                            np.array([int(self._y[k] != pred[k]) for k in range(self._X_row)]) * self._Weight)
                        if wrong_sum < best_wrong_sum:
                            best_wrong_sum = wrong_sum
                            best_pred = pred
                    else:
                        pred = np.where(self._X[:, i] >= threshold, 0, 1)
                        wrong_sum = np.sum(
                            np.array([int(self._y[k] != pred[k]) for k in range(self._X_row)]) * self._Weight)
                        if wrong_sum < best_wrong_sum:
                            best_wrong_sum = wrong_sum
                            best_pred = pred
        best_final_pred = np.where(best_pred == 0, -1, 1)
        return best_wrong_sum, best_pred, best_final_pred


if __name__ == '__main__':
    train_data, train_label = read()
    ada = Adaboost(train_data, train_label, 50)
    ada.train()
