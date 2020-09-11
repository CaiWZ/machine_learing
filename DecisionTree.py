'''
Descripttion: 李航《统计学习方法》决策树复现
version: v1.0
Author: WenZeng Cai
Date: 2020-09-10 16:59:26
LastEditors: WenZeng Cai
LastEditTime: 2020-09-11 15:28:52
'''
import numpy as np
import time

def loadData(filename):
  
    '''
    name: 加载数据
    msg: 
    param {str} 文件名
    return {list}数据集和标签集 
    ''' 
    data,label=[],[]

    fr = open(filename)

    for line in fr.readlines():
        curLine=line.strip().split(',')

        # 归一化处理
        # dataArr.append([int(int(num) > 128) for num in curLine[1:]])
        data.append([int(int(num)>128 )for num in curLine[1:]])
        label.append(int(curLine[0]))
    return data,label

def getMajorClass(trianLabel):
    '''
    name: 得到标签中最多的类
    msg: 
    param {type} 标签集
    return {type} 最大的标签
    '''
    classlabel={}
    for i in range(len(trianLabel)):
        if trianLabel[i] in classlabel.keys():
            classlabel[trianLabel[i]] += 1
        else:
            classlabel[trianLabel[i]]=1

    
    #降序排列 
    classLableSort=sorted(classlabel.items(),key=lambda x: x[1],reverse=True)
    return classLableSort[0][0]

def cal_H_D(trainLabel):
    '''
    name: 计算H(D),即数据集的经验熵
    msg: 
    param {type}标签集 
    return {type} 经验熵
    '''
    trainLabelSet = set([label for label in trainLabel])
    H_D=0
    for i in trainLabelSet:
        p=trainLabel[trainLabel==i].size/trainLabel.size
        H_D += -1 * p * np.log2(p)
    return H_D

def cal_H_D_A(trainData_DevFeature,trainLabel):
    '''
    name: 计算特征A对数据集D的经验条件熵
    msg: 
    param {type} 数据集D只有特征的切割部分
    return {type} 经验条件熵H(D|A)
    '''
    H_D_A=0
    trainDataSet=set([label for label in trainData_DevFeature])
    
    for i in trainDataSet:

        H_D_A += trainData_DevFeature[trainData_DevFeature==i].size/(trainData_DevFeature).size \
            *cal_H_D(trainLabel[trainData_DevFeature==i])
    
    return H_D_A


def calcBestFeature(trainDataList,trainLabelList):
        
    trainDataArr=np.array(trainDataList)
    trainLabelArr=np.array(trainLabelList)
    
    featureNum=trainDataArr.shape[1]

    max_G_D_A=-1
    max_feature=-1

    H_D=cal_H_D(trainLabelArr)

    for feature in range(featureNum):
        trainDataArr_DevFeature=np.array(trainDataArr[:,feature].flat)
        G_D_A=H_D-cal_H_D_A(trainDataArr_DevFeature,trainLabelArr)
        
        if G_D_A>max_G_D_A:
            max_G_D_A=G_D_A
            max_feature=feature
    return max_feature,max_G_D_A


def getSubArr(trainData,trainLabel,A,a):
    '''
    name:更新数据集和标签集
    msg: 
    param {type} 数据集，训练集,要去除的特征A，当data[A]== a时，说明该行样本时要保留的
    return {type} 更新后的数据集，训练集
    '''
    retTrainData,retTrainLabel=[],[]
    for i in range(len(trainData)):
        if trainData[i][A]==a:
            retTrainData.append(trainData[i][0:A]+trainData[i][A+1:])
            retTrainLabel.append(trainLabel[i])
    return retTrainData,retTrainLabel

def createTree(*dataSet):
    trainDataList=dataSet[0][0]
    trainLabelList=dataSet[0][1]

    Epsilon=0.1
    #打印信息：开始一个子节点创建，打印当前特征向量数目及当前剩余样本数目
    print('start a node', len(trainDataList[0]), len(trainLabelList))
    classDict={label for label in trainLabelList}
    if len(classDict)==1:
        return trainLabelList[0]

    if len(trainLabelList)==0:
        return getMajorClass(trainLabelList)

    Ag,G_D_A=calcBestFeature(trainDataList,trainLabelList)
    if G_D_A<Epsilon:
        return getMajorClass(trainLabelList)
    treeDict={Ag:{}}
    treeDict[Ag][0]=createTree(getSubArr(trainDataList,trainLabelList,Ag,0))
    treeDict[Ag][1]=createTree(getSubArr(trainDataList,trainLabelList,Ag,1))
    return treeDict

def predict(testData,tree):
    while True:
        (key,value),=tree.items()
        if type(tree[key]).__name__=='dict':
            dataVal=testData[key]
            del testData[key]
            tree=value[dataVal]
            if type(tree).__name__=='int':
                return tree
        else:
            return value

def test(testData,testLabel,tree):
    errCnt=0
    for i in range(len(testData)):
        if testLabel[i] != predict(testData[i],tree):
            errCnt += 1
    return 1-errCnt/len(testData)

if __name__=='__main__':
    #开始时间
    start = time.time()

    # 获取训练集
    trainDataList, trainLabelList = loadData('../Mnist/mnist_train.csv')
    # 获取测试集
    testDataList, testLabelList = loadData('../Mnist/mnist_test.csv')

    #创建决策树
    print('start create tree')
    tree = createTree((trainDataList, trainLabelList))
    print('tree is:', tree)

    #测试准确率
    print('start test')
    accur = test(testDataList, testLabelList, tree)
    print('the accur is:', accur)

    #结束时间
    end = time.time()
    print('time span:', end - start)

                
        
    
    

    
    
    



    


        
    
    
    


       
