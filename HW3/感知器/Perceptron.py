import random
from numpy import *
import numpy as np
import matplotlib.pyplot as plt


def Plot(X, Y, theta):
    index1 = []
    index0 = []
    for i in range(len(Y)):
        if Y[i] == 1:
            index1.append(i)
        else:
            index0.append(i)
    x1 = []
    x0 = []
    y1 = []
    y0 = []
    for i in index1:
        x1.append(X[i][1])
        y1.append(X[i][2])
    for i in index0:
        x0.append(X[i][1])
        y0.append(X[i][2])
    plt.scatter(x1, y1, color='red')
    plt.scatter(x0, y0, color='blue')
    if theta[2][0] != 0:
        x2 = np.arange(-5, 5, 0.1)
        y2 = (float(theta[0][0]) + float(theta[1][0]) * x2)/float(-theta[2][0])
        plt.plot(x2, y2, color='yellow')
    else:
        plt.axvline(x=float(-(theta[0][0]/theta[1][0])), ls='--', c='red')

    plt.show()


def Perceptron(X, label):
    n = 1                                     # 学习率这里取1
    theta = np.array([[0], [0], [0]])         # theta为2x1矩阵
    wrong_number = 0                        # 定义出错数量
    for i in range(len(X)):
        y_hat = np.dot(X[i], theta)          # 求取第i个样本的加权和
        y = y_hat
        if y_hat >= 0:
            y_hat = 1                      # 此处为sign函数
        else:
            y_hat = -1
        if y_hat != label[i]:               # 如果预测值与实际值不相等，做出如下修正
            theta[0][0] += label[i]*n
            theta[1][0] += (label[i])*X[i][1]*n
            theta[2][0] += (label[i])*X[i][2]*n
            wrong_number += 1               # 出错数量加1
    return theta,wrong_number


X1 = np.random.randint(-10, 10, size=(50, 3))  # 定义一个50x3的矩阵，作为增广特征矩阵
X2 = np.array(ones((len(X1), 3)))            # 定义一个和增广特征矩阵维度相同的矩阵，用来储存打乱顺序之前的特征矩阵
for i in range(len(X1)):
    for j in range(len(X1[0])):
        X2[i][j] = X1[i][j]
np.random.shuffle(X1)                       # 打乱顺序的过程
for i in range(len(X1)):
    X1[i][0] = 1
    X2[i][0] = 1                            # 将增广特征矩阵每个向量的第一维度置1
Y1 = np.array(ones((len(X1))))
Y2 = np.array(ones((len(X1))))
for i in range(len(X1)):                    # 设置label数值
    if X1[i][1] > 0.5:
        Y1[i] = 1
    else:
        Y1[i] = -1
for i in range(len(X1)):
    if X2[i][1] > 0.5:
        Y2[i] = 1
    else:
        Y2[i] = -1
print(Y1, Y2)
print(X1)
print(X2)
theta1, w1 = Perceptron(X1, Y1)
theta2, w2 = Perceptron(X2, Y2)
print('S mistake:', w1)
print('------------------------------')
print('S\' mistake:', w2)
Plot(X1, Y1, theta2)
