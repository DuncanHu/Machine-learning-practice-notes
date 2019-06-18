import operator
import numpy as np
from numpy import *

def str_3(str_i):
    if str_i=='largeDoses':
        return 3
    elif str_i=='smallDoses':
        return 2
    elif str_i=='didntLike':
        return 1
    else:
        return np.nan

def file2matrix(filename):
    fr = open(filename,'r')
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return
    fr2 = open(filename,'r+')
    index = 0

    for line in fr2.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3] #循环给每行传输特征
        classLabelVector.append(str_3(listFromLine[-1])) #每个行数据的label
        index += 1
    return returnMat,classLabelVector

# import matplotlib
# import matplotlib.pyplot as plt
# fig=plt.figure()
# ax=fig.add_subplot(111)
# ax.scatter(returnMat[:,1],returnMat[:,2],15.0*array(classLabelVector),15.0*array(classLabelVector))

# plt.show()

# 数值归一化公式
# newValue = (oldvalue-min) / (max-min)
def autoNorm(dataSet):
    #获得数据的最小值 按列寻找最大|最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    # print('minVals:\n',minVals)
    # print('maxVals:\n',maxVals)
    #最大值和最小值的范围
    ranges = maxVals - minVals
    #shape(dataSet)返回dataSet的矩阵行列数
    normDataSet = np.zeros(np.shape(dataSet))
    #返回dataSet的行数
    m = dataSet.shape[0]
    #原始值减去最小值
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    #除以最大和最小值的差,得到归一化数据
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    #返回归一化数据结果,数据范围,最小值
    return normDataSet, ranges, minVals

# classify0(测试集特征，训练集特征，训练集label，4)
def classify0(inX, dataSet, labels, k):
    #numpy函数shape[0]返回dataSet的行数
    dataSetSize = dataSet.shape[0]
    # 在列向量方向上重复inX共1次(横向),行向量方向上重复inX共dataSetSize次(纵向)
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    print(diffMat)
    print('==================================\n==================================\n==================================\n')
    # 二维特征相减后平方
    sqDiffMat = diffMat**2
    # sum()所有元素相加,sum(0)列相加,sum(1)行相加
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方,计算出距离
    distances = sqDistances**0.5
    # 返回distances中元素从小到大排序后的索引值
    sortedDistIndices = distances.argsort()
    # 定一个记录类别次数的字典
    classCount = {}
    for i in range(k):
        # 取出前k个元素的类别
        voteIlabel = labels[sortedDistIndices[i]]
        # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        # 计算类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    # python3中用items()替换python2中的iteritems()
    # key=operator.itemgetter(1)根据字典的值进行排序
    # key=operator.itemgetter(0)根据字典的键进行排序
    # reverse降序排序字典
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    # 返回次数最多的类别,即所要分类的类别
    return sortedClassCount[0][0]


def datingClassTest():
    #打开的文件名
    filename = "data/datingTestSet.txt"
    #将返回的特征矩阵和分类向量分别存储到datingDataMat和datingLabels中
    datingDataMat, datingLabels = file2matrix(filename)
    #取所有数据的百分之十
    hoRatio = 0.10
    #数据归一化,返回归一化后的矩阵,数据范围,数据最小值
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #获得normMat的行数 （特征矩阵）
    m = normMat.shape[0]
    #百分之十的测试数据的个数
    numTestVecs = int(m * hoRatio)
    #分类错误计数
    errorCount = 0.0

    for i in range(numTestVecs):
        # 前numTestVecs个数据作为测试集,后m-numTestVecs个数据作为训练集
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],
            datingLabels[numTestVecs:m], 4)
        print("分类结果:%d\t真实类别:%d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("错误率:%f%%" %(errorCount/float(numTestVecs)*100))


if __name__ == '__main__':
    datingDataMat, datingLabels = file2matrix('data/datingTestSet.txt')
    # print(datingDataMat)
    # print(datingLabels)
    datingDataMat, ranges,minVals = autoNorm(datingDataMat)
    # print('datingDataMatL:\n',datingDataMat)
    # print('datingLabels:\n',datingLabels)
    # print(minVals)
    datingClassTest()
