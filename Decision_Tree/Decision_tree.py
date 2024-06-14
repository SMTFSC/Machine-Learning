import numpy as np
import matplotlib.pyplot as plt


# 采用信息熵和信息增益来构建决策树(ID3)
# 也可以采用基尼系数和基尼增益来构建
# 首先计算总的数据集的信息熵
# 然后对于每一个特征分别计算信息增益，选取有最大增益的那个特征进行划分数据集

class Decision_Tree:
    def __init__(self, X, y, features, train_data_names, y_label, X_label):
        self._X = X
        self._y = y
        self._train_data_names = train_data_names
        self._y_label = y_label
        self._X_label = X_label
        self._y_len = len(self._y)
        self._dataSet = np.hstack((self._X, self._y.reshape(self._y_len, 1)))
        self._features = features

    def choose_best_feature(self, dataSet, features):
        base_entropy = self.calculate_entropy(dataSet[:, -1])
        best_feature = -1
        # 防止返回-1,考虑极限情况,当entropy与base_entropy相等时,
        # 当然这个值可以随便取,只要是负的就行
        best_info_gain = -0.001
        for feature in features:
            entropy = 0
            indexes = np.unique(dataSet[:, feature])
            for index in indexes:
                temp = self.split_dataSet(dataSet, feature, index)
                count = len(temp)
                entropy += count / len(dataSet) * self.calculate_entropy(temp[:, -1])
            # 假设这里只剩下一个特征,那么entropy = base_entropy,那么应该返回index而不是-1,不然会报错
            if base_entropy - entropy > best_info_gain:
                best_feature = feature
                best_info_gain = base_entropy - entropy
        return best_feature

    def calculate_entropy(self, array):
        # 注意这个地方不能用np.bincount来计算熵,不然会出现log(0)
        indexes = np.unique(array)
        entropy = 0
        for index in indexes:
            count = np.sum(array == index)
            temp = count / len(array)
            entropy += -np.log(temp) * temp
        return entropy

    def train(self):
        result_dict = self.create_tree(self._dataSet, self._features)
        print(result_dict)

    def split_dataSet(self, dataSet, feature_num, value):
        return dataSet[dataSet[:, feature_num] == value]

    def create_tree(self, dataSet, features):
        # 如果无特征可选,那么就直接返回y中最多的那个
        if len(features) == 0:
            return self.majority_count(dataSet[:, -1])
        # 如果label都一样,直接返回,停止迭代
        if len(np.unique(dataSet[:, -1])) == 1:
            return dataSet[0][-1]
        best_feature = self.choose_best_feature(dataSet, features)
        result_dict = {}
        indexes = np.unique(dataSet[:, best_feature])
        # 这里添加特征时显得很臃肿，可以考虑把数字特征重新转为字符串
        result_dict[self._X_label[best_feature]] = {}
        for index in indexes:
            features_copy = features.copy()
            new_dataSet = self.split_dataSet(dataSet, best_feature, index)
            features_copy.remove(best_feature)
            # train_data_names[best_feature][index]表达的是根据best_feature划分时的各个类别
            # self._X_label[best_feature]所选特征的名字,主要是为了可视化
            result_dict[self._X_label[best_feature]][self._train_data_names[best_feature][index]] = \
                self.create_tree(new_dataSet, features_copy)
        return result_dict

    # 可视化决策树
    # 图就不画了，字典对了就行，画图太浪费时间
    def plot_tree(self):
        pass

    def majority_count(self, label_list):
        # 当np.bincount(label_list)有多个值一样时,会返回下标最小的那个
        return np.argmax(np.bincount(label_list))


# 这里的feature转成数字只针对文本分类,如果是数字的话要重新考虑
# 其实转成数字也没必要，反而后面的处理变麻烦了
def feature_to_number(array):
    temp_dict = {}
    flag = 0
    for data in array:
        if data not in temp_dict:
            temp_dict[data] = flag
            flag += 1
    result = []
    for data in array:
        result.append(temp_dict[data])
    return np.array(result), list(temp_dict.keys())


if __name__ == '__main__':
    with open('lenses.txt') as f:
        lines = f.readlines()
        train_data = []
        train_label = []
        for line in lines:
            temp_array = line.strip().split('\t')
            train_data.append(temp_array[:-1])
            train_label.append(temp_array[-1])
        train_num_label, _ = feature_to_number(train_label)
        # 当y为数字时对应的标签
        y_label = {0: 'no lenses', 1: 'soft', 2: 'hard'}
        # 当特征为数字时对应的特征
        X_label = {0: 'age', 1: 'prescript', 2: 'astigmatic', 3: 'tearRate'}
        # 注意这里要加一个dtype = int,不然后面类型会出错, 因为zeros默认是float64
        train_array = np.zeros((len(train_data), len(train_data[0])), dtype=int)
        train_data_names = []
        for i in range(len(train_data[0])):
            train_data_line_i = [train_data[j][i] for j in range(len(train_data))]
            train_array[:, i], data_name = feature_to_number(train_data_line_i)
            train_data_names.append(data_name)
    # [0, 1, 2, 3]可以写成list(range(len(train_data[0])))便于泛化，为了理解方便还是写成
    # [0, 1, 2, 3]
    d = Decision_Tree(train_array, train_num_label, [0, 1, 2, 3], train_data_names, y_label, X_label)
    d.train()
