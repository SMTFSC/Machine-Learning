import numpy as np


# 计算支持度和可信度
# 分别对应频繁项集和关联规则
# Apriori算法的关键在于一个原理:
# 频繁是指支持度大于最小支持度
# 如果一个项集是频繁的,那么这个项集的子集也是频繁的
# 也就是说,如果某个项集是不频繁的,那么他的父集也是不频繁的
# 利用上述规律可以简化计算(重点)
class Apriori:
    def __init__(self, X, min_support):
        self._X = X
        self._X_row = self._X.shape[0]
        self._min_support = min_support
        self._set_num = 0
        self._record_dict = {}

    def train(self):
        # 首先选出项集
        pass

    def get_set(self, num, left_list):
        pass


if __name__ == '__main__':
    train_data = [np.array([1, 3, 4]),
                  np.array([2, 3, 5]),
                  np.array([1, 2, 3, 5]),
                  np.array([2, 5]),
                  ]
    train_data = np.array(train_data, dtype=object)
    a = Apriori(train_data, 0.5)
    a.train()
