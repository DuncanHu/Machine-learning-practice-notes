'''
机器学习实战——KNN基础代码实现

'''

from numpy import *
import numpy as np
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]]) # 特征矩阵
    labels = ['A', 'A', 'B', 'B'] # label向量
    return group, labels

# 参数：inX：测试数据 dataSet：特征矩阵 labels：label向量 k：k个近邻
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] #获取特征矩阵的行数
    # 计算欧式距离（两种方法）
    # 方法一
    # diffMat = tile(inX,(dataSetSize,1)) - dataSet
    # sqDiffMat = diffMat**2
    # sqDistances = sqDiffMat.sum(axis=1)
    # distances = sqDistances ** 0.5
    # 方法二
    diffMat = inX - dataSet # 矩阵的广播机制
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5

    # 对计算出的距离进行排序，以得到最近的点
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 # get(value，default) value:返回此键的值，default:如果此键不存在返回0
    # 根据字典的值对字典进行排序
    sortedClassCount = sorted(classCount.items(), key = lambda item:item[1], reverse = True)
    print(sortedClassCount)
    return sortedClassCount[0][0]

group, labels=createDataSet()

result=classify0(np.array([[10,0]]),group,labels,3)
print(result)