'''
Descripttion: 
version: 
Author: WenZeng Cai
Date: 2020-09-08 16:04:39
LastEditors: WenZeng Cai
LastEditTime: 2020-09-09 16:58:38
'''
import numpy as np
import time

def loadData(filename):
    
    '''
    name: 读取数据
    msg: 
    param {type} 文件名
    return {type} 数据和标签
    '''  
    print('加载数据')  
    fr =  open(filename)

    data,label=[],[]

    for line in fr.readlines():
        curLine=line.strip().split(',')

        # 对数据进行二值化处理，当大于128时取1，否则取0
        data.append([int(int(num) > 128) for num in curLine[1:]])
        label.append(int(curLine[0]))
    return data,label
    
def NavieBayes(Py,Px_y,x):
  
    '''
    name: 由朴素贝叶斯得到 概率分布
    msg: 
    param {type} 先验概率，条件概率，样本数据
    return {type} 所有label的概率
    '''
    # 数据集图片为28 * 28的，向量化就为784维
    featureNum=784
    # 共有0-9个类别
    classNum=10
    # 所有标记的概率数组
    P=[0]*classNum
    
    # 对每个类别进行遍历，书中的式子使用的是相乘，为了避免出现零过多而下溢出的情况
    # 使用了对数处理，而对数处理相乘就可以转化为相加。
    for i in range(classNum):
        sum=0
        # 获取每个条件概率值，进行相加
        for j in range(featureNum):
            sum += Px_y[i][j][x[j]]
        # 最后计算每个类别，使用先验概率和条件概率相加。(由于对数处理，操作都变为相加)
        P[i]=sum + Py[i]
    # 得到类别最大的概率的索引
    return P.index(max(P))



def getProbability(trainData,trainLabel):
    '''
    name: 由训练集获得先验概率和条件概率
    msg: 
    param {type} 训练数据和便签数据
    return {type} 先验概率py,条件概率px_y
    '''
    # 数据集数据为28 * 28的图片，所以转换为784维向量
    featureNum=784
    
    # 共有0-9个类别
    classNum=10
    
    # 初始化先验概率为10 * 1矩阵
    Py=np.zeros((classNum,1))

    for i in range(classNum):
        # 这个式子较为复杂，分开解析，首先将trainLabel转换为矩阵，并与i进行比较。
        # 使用sum函数统计上述为True，即相等情况下的数目，使用贝叶斯估计，对分子加1
        # 那么分母就需要相应的加K，即10，求得先验概率
        Py[i]=((np.sum(np.mat(trainLabel)==i))+1)/(len(trainLabel)+10)
    # 最后对先验概率进行对数处理
    Py=np.log(Py)

    # 计算条件概率
    Px_y=np.zeros((classNum,featureNum,2))
    
    # 计算书中公式4.10分子部分，首先计算分子部分
    # 对标签集进行遍历
    for i in range(len(trainLabel)):
        # 当前标签
        label=trainLabel[i]
        # 当前数据
        x=trainData[i]
        # 遍历每个特征
        for j in range(featureNum):
            # 对每个特征进行累加处理
            Px_y[label][j][x[j]] +=1
    

    # 计算公式4.10的分母以及最终结果
    # 遍历类别共十种
    for label in range(classNum):
        # 遍历特征数
        for j in range(featureNum):
            # 得到当前标签，第j个特征为0的个数
            Px_y0=Px_y[label][j][0]
            # 得到当前标签，第j个特征为1的个数
            Px_y1=Px_y[label][j][1]
            # 公式4.10的除法计算，依据贝叶斯公式，且采用对数处理
            # 计算当前标签，数据x的第j个特征为0和1的条件概率
            Px_y[label][j][0]=np.log((Px_y0+1)/(Px_y0+Px_y1+2))
            Px_y[label][j][1]=np.log((Px_y1+1)/(Px_y0+Px_y1+2))
    return Py,Px_y

def test(testData,testLabel,Py,Px_y):
    '''
    name: 测试集数据
    msg: 
    param {type} 测试数据，测试标签，先验概率，条件概率
    return {type} 正确率
    '''
    print()
    errCnt=0
    
    for i in range(len(testData)):
        # 计算概率
        pro=NavieBayes(Py,Px_y,testData[i])
        if pro!=testLabel[i]:
            # 当计算和实际不相符时，错误加1
            errCnt +=1
    return 1-(errCnt/len(testData))

if __name__ == "__main__":
    start = time.time()
    # 获取训练集
    print('start read transSet')
    trainDataArr, trainLabelArr = loadData('../Mnist/mnist_train.csv')

    # 获取测试集
    print('start read testSet')
    testDataArr, testLabelArr = loadData('../Mnist/mnist_test.csv')

    #开始训练，学习先验概率分布和条件概率分布
    print('start to train')
    Py, Px_y = getProbability(trainDataArr, trainLabelArr)

    # print('Pxy',Px_y)

    #使用习得的先验概率分布和条件概率分布对测试集进行测试
    print('start to test')
    accuracy = test(testDataArr, testLabelArr,Py, Px_y)

    #打印准确率
    print('the accuracy is:', accuracy)
    #打印时间
    print('time span:', time.time() -start)
      

    

    