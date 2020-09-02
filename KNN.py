'''
Descripttion: 李航《统计学习方法》K邻近原理
version: v1.0
Author: WenZeng Cai
Date: 2020-09-02 10:22:47
LastEditors: WenZeng Cai
LastEditTime: 2020-09-02 12:27:47
'''
import numpy as np
import time
def loadData(filename):
    '''
    name: 加载数据
    msg: 
    param {type} 文件名
    return {type} 数据集和标签
    '''
    print('加载数据')
    data,label=[],[]

    fr =  open(filename)

    for i in fr.readlines():
        curLine=i.strip().split(',')

        data.append([int(num) for num in curLine[1:]])
        label.append(int(curLine[0]))
    return data,label

def calDist(x1,x2):
    
    '''
    name: 计算两个向量欧氏距离
    msg: 
    param {type} 两个向量
    return {type} 距离
    ''' 
    print('计算距离')
    return np.sqrt(np.sum(np.square(x1-x2)))

def getCloset(trainData,trainLabel,x,topK):
    '''
      name: 得到样本的标签
      msg: 通过获取样本最近的topK点，数目最多的那个为标签
      param {type} 训练数据，训练标签，样本点，最近点的数目
      return {type} 样本的标签
      '''
    print('计算标签')
    n=len(trainLabel)
    dist=[0]*n
    for i in range(len(trainData)):
        x2=trainData[i]
        y=calDist(x2,x)
        dist.append(y)

    #argsort返回的是升序的标签，我们取前k个最小的 
    topKlist= np.argsort(dist)[:topK]

    labelList=[0]*20
    
    for index in topKlist:
        # 计算数量，返回最多的
        labelList[int(trainLabel[index])]+=1
    return labelList.index(max(labelList))

def test(trainData,trainLable,testData,testLabel,topK):
    print('测试')
    trainDataMat = np.mat(trainData); trainLabelMat = np.mat(trainLable).T
    testDataMat = np.mat(testData); testLabelMat = np.mat(testLabel).T
    
    errCnt=0
    
    for i in range(300):
        x=testDataMat[i]
        print('测试集形状',np.shape(trainDataMat))
        y=getCloset(trainDataMat,trainLabelMat,x,topK)
        if y!=testLabelMat[i]:
            errCnt+=1
    return 1-(errCnt/300)


if __name__ == "__main__":
    start = time.time()

    #获取训练集
    trainDataArr, trainLabelArr = loadData('../Mnist/mnist_train.csv')
    #获取测试集
    testDataArr, testLabelArr = loadData('../Mnist/mnist_test.csv')
    #计算测试集正确率
    print('主函数',np.shape(trainDataArr))
    accur = test(trainDataArr, trainLabelArr, testDataArr, testLabelArr, 25)
    #打印正确率
    print('accur is:%d'%(accur * 100), '%')

    end = time.time()
    #显示花费时间
    print('time span:', end - start)
        
    
    
        
    
      


    


