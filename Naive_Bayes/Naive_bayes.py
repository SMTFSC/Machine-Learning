import numpy as np


# 1.naive_bayes 假设所有特征是独立的
# 2.考虑到p(w)都是一样的,所以没必要计算,只需要计算p(w|ci)*p(ci)即可,计算出正类的概率和负类的概率
# 3.考虑到要预测的文本可能会不存在与vocabulary_list中(如负例中的词不会出现在正例中),那样的话就无法计算概率,
# 所以可以把所有的词初始化为1次，避免概率为零的情况
# 4.该类中使用的是词集模型(set_of_words model), 即以词在文档中是否出现作为特征,就是说如果该词出现了,
# 就只能出现一次(在word_to_vector中使用的是 '= 1' 操作 ),但事实上如果一个词在文档中出现多次可以表达出额外的信息
# 5.词袋模型(bag-of-words model)将一个词在文档中出现的次数作为特征,反映在word-to-vector中就是 ' += 1'操作

class Naive_Bayes:
    def __init__(self, posting_list, class_vector, model='SOW'):
        """

        :param posting_list:
        :param class_vector:
        :param model: 'SOW' 就是 set-of-words model  'BOW' 就是 bag-of-words model
        """
        self._posting_list = posting_list
        self._vocabulary_list = self.create_vocabulary_list()
        self._class_vector = np.array(class_vector)
        self._dataSet = np.array(self.create_dataSet())
        self._pos_part = None
        self._neg_part = None
        # 计算p(ci)
        self._pos_rate = self._class_vector[self._class_vector == 1].size \
            / self._class_vector.size
        self._neg_rate = self._class_vector[self._class_vector == 0].size \
            / self._class_vector.size

    def forward(self):
        # 选取出正例
        pos_index = self._class_vector == 1
        pos_part = self._dataSet[pos_index]
        # 选取出负例
        neg_index = self._class_vector == 0
        neg_part = self._dataSet[neg_index]
        # 计算p(w|ci)
        # 这里加一的目的是为了避免有概率为零导致乘积为0的情况
        pos_part = np.sum(pos_part, axis=0) + 1
        self._pos_part = pos_part / (np.sum(pos_part) + 2)
        neg_part = np.sum(neg_part, axis=0)
        self._neg_part = neg_part / (np.sum(neg_part) + 2)

    def predict(self, feature_list):
        feature_vector = self.word_to_vector(feature_list)
        # 这里加上1e5的目的是防止log中的数为零,乘积为0的情况出现在要预测的文本一个
        # word都不在vocabulary_list中, 在这种情况下数量多的用例取胜(不是很合理)
        pos_possibility = np.sum(np.log(self._pos_part * feature_vector + 1e5)) + np.log(self._pos_rate)
        neg_possibility = np.sum(np.log(self._neg_part * feature_vector + 1e5)) + np.log(self._neg_rate)
        if pos_possibility > neg_possibility:
            print('1')
        elif pos_possibility < neg_possibility:
            print('0')
        else:
            print('unknown')

    def train(self):
        self.forward()

    def create_vocabulary_list(self):
        voca_set = set()
        for post in self._posting_list:
            voca_set = voca_set | set(post)
        return list(voca_set)

    def word_to_vector(self, posting_vector):
        result_vec = [0] * len(self._vocabulary_list)
        for post in posting_vector:
            if post in self._vocabulary_list:
                result_vec[self._vocabulary_list.index(post)] = 1
        return result_vec

    def create_dataSet(self):
        data_Set = []
        for post in self._posting_list:
            result_vec = self.word_to_vector(post)
            data_Set.append(result_vec)
        return data_Set


if __name__ == '__main__':
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    n = Naive_Bayes(postingList, classVec)
    n.train()
    n.predict(feature_list=['love', 'my', 'dalmation'])
    n.predict(feature_list=['stupid', 'garbage'])
