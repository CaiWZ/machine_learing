'''
Descripttion: 李航《统计学习方法》逻辑斯蒂复现
version: v1.0
Author: WenZeng Cai
Date: 2020-09-12 15:22:13
LastEditors: WenZeng Cai
LastEditTime: 2020-09-12 15:56:46
'''
import numpy as np
import time

def loadData(filename): 
    '''
    name: 加载数据
    msg: 
    param {type} 文件名
    return {type} 数据和标签
    '''
    data,label=[],[]
    fr=open(filename)
    for line in fr.readlines():
        curLine=line.strip().split(',')
        # 二分法，以0为界限，划分两类标签
        if int(curLine[0])==0:
            label.append(1)
        else:
            label.append(0)
        # 对数据进行归一化处理
        data.append([int(num)/255 for num in curLine[1:]])

    return data,label

def logisticRegression(trainData,trainLabel,iter=200):

    '''
    name: 逻辑斯蒂回归训练过程
    msg: 
    param {type} 数据集，训练集，迭代次数
    return {type} 训练所得w
    '''

    #按照书本“6.1.2 二项逻辑斯蒂回归模型”中式6.5的规则，将w与b合在一起，
    #此时x也需要添加一维，数值为1
    #循环遍历每一个样本，并在其最后添加一个1
    for i in range(len(trainData)):
        trainData[i].append(1)
    
    trainData=np.array(trainData)
    w=np.zeros(trainData.shape[1])
    h=0.001
    for i in range(iter):
        for j in range(trainData.shape[0]):
            wx=np.dot(w,trainData[j])
            xi=trainData[j]
            yi=trainLabel[j]
            # 根据似然函数的求导公式得到梯度上升更新w
            w += h*(yi*xi-(np.exp(wx)*xi)/(1+np.exp(wx)))
    return w
    
def precit(w,x):
    '''
    name: 预测分类
    msg: 
    param {type} 偏差w，待预测值
    return {type}预测结果
    '''
    wx=np.dot(w,x)
    pro=(np.exp(wx))/(1+np.exp(wx))
    if pro>0.5:
        return 1
    else:
        return 0

def test(testData,testLabel,w):
    '''
    name: 测试
    msg: 
    param {type} 测试集，测试标签集，偏差w
    return {type} 正确率
    '''
    errCnt=0

    for i in range(len(testData)):
        testData[i].append(1)
    
    for i in range(len(testData)):
        if testLabel[i]!=precit(w,testData[i]):
            errCnt+=1
    return 1-errCnt/len(testData)

if __name__=='__main__':
    start = time.time()

    # 获取训练集及标签
    print('start read transSet')
    trainData, trainLabel = loadData('../Mnist/mnist_train.csv')

    # 获取测试集及标签
    print('start read testSet')
    testData, testLabel = loadData('../Mnist/mnist_test.csv')

    # 开始训练，学习w
    print('start to train')
    w = logisticRegression(trainData, trainLabel)

    #验证正确率
    print('start to test')
    accuracy = test(testData, testLabel, w)

    # 打印准确率
    print('the accuracy is:', accuracy)
    # 打印时间
    print('time span:', time.time() - start)

    


