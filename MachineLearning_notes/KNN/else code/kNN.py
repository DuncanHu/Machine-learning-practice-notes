from numpy import *
import operator
from os import listdir

# training samples
sample = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])

# the labels of samples
label = ['A', 'A', 'B', 'B']

def classify(data, sample, label, k):
    SampleSize = sample.shape[0] # 训练样本集的行数
    # DataMat = tile(data, (SampleSize, 1))  #将data扩展到和训练样本集sample一样的行数
    #计算欧式距离
    # delta = (DataMat - sample)**2
    delta = (data - sample)**2 #利用广播机制相减
    distance = (delta.sum(axis = 1))**0.5  # 以上三步计算欧氏距离

    sortedDist = distance.argsort()  # 对欧氏距离向量进行排序 argsort返回数值从小到大的索引值
    classCount = {}

    # 以下操作获取距离最近的k个样本的标签
    for i in range(k):
        votedLabel = label[sortedDist[i]]
        classCount[votedLabel] = classCount.get(votedLabel, 0) + 1 #get(返回此键的值，default) 返回指定键的值,如果此键不存在返回0
    #两种方法
    # result = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    result = sorted(classCount.items(), key = lambda item:item[1], reverse = True)
    return result[0][0]

print(classify(np.array([[10,0]]), sample, label, 3)) # test
