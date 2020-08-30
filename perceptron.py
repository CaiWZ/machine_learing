'''
Descripttion: 李航《统计学习方法》复现，感知机
version: 
Author: WenZeng Cai
Date: 2020-08-29 15:50:51
LastEditors: WenZeng Cai
LastEditTime: 2020-08-30 10:57:51
'''
import numpy as np
import time


def loadData(filename:str)->list:
  '''
  name: 
  msg: 
  param {type} 
  return {type} 
  '''   
  print('加载数据')
#   数据和标签
  data,label=[],[]

  fr=open(filename,'r')

  for line in fr.readlines():
    #   每行以逗号分隔
      cur=line.strip().split(',')


    #   数据集有0-9标签，0-5标记为1，大于标记为-1，二分
      if int(cur[0])>=5:
          label.append(1)
      else:
          label.append(-1)
    #  除第一列外，进行归一化处理
      data.append([int(num)/255 for num in cur[1:]])


  return data,label


def perceptron_train(data:list,label:list,iter=50)->int:
    '''
      name: 感知训练
      msg: 
      param {type} 数据和标签,默认迭代50次
      return {type}训练好的超平面参数w和b 
      '''

    print('开始训练')

    # 转化为矩阵，label转置,label本题是1*N列表，可以直接使用label[i]获得标签数据
    # 为了格式统一将他转置为矩阵，为了使用label[i]得到数据，就需要转置为列矩阵
    dataMat=np.mat(data)
    labelMat=np.transpose(np.mat(label))

    # 取数据维度
    m,n=np.shape(dataMat)

    # 初始化超平面参数，和偏置量为0，使用随机梯度下降，步长为0.001
    w=np.zeros((1,n))
    b=0
    h=0.0001

    for k in range(iter):

        for i in range(m):
            xi=dataMat[i]
            yi=labelMat[i]
            #   -yi(w*xi+b)>=0
            # 误分类部分为>=0,进行梯度下降更新w和b
            if -yi * (w * xi.T + b) >= 0:
                w = w + h *  yi * xi
                b = b + h * yi
    return w,b

def test(data,label,w,b):
    '''
    name: 模型测试
    msg: 
    param {type} 数据测试集，标签，权重w,偏置b
    return {type} 精确度
    '''
    print('开始测试')
    dataMat=np.mat(data)
    labelMat=np.mat(label).T

    errCnt=0

    m,n=np.shape(dataMat)

    for i in range(m):
        xi=dataMat[i]
        yi=labelMat[i]
        # 对测试集进行计算，当>=0时就是误判加1。
        result=-yi * (w * xi.T + b)
        if result>=0:
            errCnt+=1
    # 得到正确率
    accuracy=1-(errCnt/m)
    return accuracy

      

if __name__ == '__main__':
    start=time.time()

    trainData,trainLabel=loadData('../Mnist/mnist_train.csv')  
    testData,testLabel=loadData('../Mnist/mnist_test.csv') 

    w,b=perceptron_train(trainData,trainLabel)
    accuracy=test(testData,testLabel,w,b)

    print('准确率',accuracy)
    end=time.time()
    print('运行时间',end-start)

