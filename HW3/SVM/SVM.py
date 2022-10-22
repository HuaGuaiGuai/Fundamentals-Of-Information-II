import math
from sklearn.svm import SVC
import random
from mlxtend.plotting import plot_decision_regions
import numpy as np
import matplotlib.pyplot as plt
from numpy import *


def Gauss_Kernel(X):
    """
    Gauss_Kernel
    函数功能：实现高斯核函数

    输入参数：向量化样本矩阵

    输出：新的特征向量，维度为样本数量
    """
    F = np.array(zeros((len(X), len(X))))                   # 定义一个方阵，维度为样本数量
    for i in range(len(X)):
        for j in range(len(X)):
            B = 0                                           # 用于存储中间结果
            for a in range(len(X[0])):
                B -= ((X[i][a]-X[j][a])**2)/2               # 对于高斯核函数，方差值选取为1
            F[i][j] = math.e**(B)
    return F


def Gauss_random(mean1, cov1, mean2, cov2, number):
    """
    Gauss_random
    函数功能：生成二维高斯随机点

    输入参数：协方差矩阵和均值

    输出：特征向量和标签
    """
    np.random.seed(12)
    x1 = np.random.multivariate_normal(mean1, cov1, number)
    label1 = np.array(np.array(ones((number, 1))))
    x2 = np.random.multivariate_normal(mean2, cov2, number)
    label2 = np.array(np.array(ones((number, 1))))
    for i in range(len(label2)):
        label2[i][0] = -1
    return x1, label1, x2, label2


def Plot(x1, x2, y1, y2):
    x = []
    y = []
    for i in range(len(x1)):
        x.append(x1[i])
        y.append(y1[i][0])
    for i in range(len(x2)):
        x.append(x2[i])
        y.append(y2[i][0])
    x11 = []
    x12 = []
    x21 = []
    x22 = []
    for i in range(len(y)):
        if y[i] > 0:
            x11.append(x[i][0])
            x12.append(x[i][1])
        else:
            x21.append(x[i][0])
            x22.append(x[i][1])
    plt.scatter(x11, x12, color='red')
    plt.scatter(x21, x22, color='blue')
    plt.show()


def Predict(x, b, w):
    return x.dot(w)+b


def Pegasos(F, Y, lamda):
    n = 100000                                      # 迭代次数
    row, col = np.shape(F)
    w = np.array(ones((len(F[0]), 1)))              # 定义w矩阵初始值
    b = 0                                           # 定义b初始值
    for i in range(1, n+1):
        r = random.randint(0, row-1)                # 随机梯度下降时，随机选取一个样本的梯度
        eta = 1.0/(lamda*i)                         # 定义学习率，随着学习次数增加，学习率逐渐下降
        predict = Predict(F[r], b, w)                 # 针对随机选中的样本进行预测
        if Y[r][0]*predict[0] > 1:                   # 进行梯度优化
            w = w - w/i
        else:
            w = w - w/i
            for j in range(len(F[0])):
                w[j][0] -= eta*Y[r][0] * F[r][j]
            b += eta*Y[r][0]
    return w, b                                     # 返回最终迭代后的w和b


mean = np.array([0, 0])
cov1 = np.array([[1, 0], [0, 1]])
cov2 = np.array([[5, 0], [0, 5]])
x1, y1, x2, y2 = Gauss_random(mean, cov1, mean, cov2,50)
x = np.vstack((x1, x2))
y = np.vstack((y1, y2))
F = Gauss_Kernel(x)
print(np.shape(F))
w, b = Pegasos(F, y, 0.1)
wrong = 0
for i in range(len(F)):
    if Predict(F[i], b, w)[0] < 0:
        wrong += 1
print('正确率为：', 1-(wrong-50)/100)
print(Pegasos(F, y, 0.1))
Plot(x1, x2, y1, y2)
# 此部分为绘图部分，撰写报告时可以使用。
# x1 = []
# y1 = []
# for i in range(len(y)):
#     y1.append(int(y[i][0]))
# y2=np.array(y1)
# svm = SVC(kernel='rbf', random_state=0, gamma=1, C=100)
# svm.fit(x, y2)
# plot_decision_regions(x, y2, clf = svm)
# plt.legend(loc='upper left')
# plt.title('gamma = 1,C=10')
# plt.show()
