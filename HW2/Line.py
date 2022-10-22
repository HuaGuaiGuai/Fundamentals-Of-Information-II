import numpy as np
import matplotlib.pyplot as plt
from numpy import *


def close_formed(x, y):
    xt = x.T
    a = np.dot(np.linalg.inv(np.dot(xt, x)), np.dot(xt, y))
    return a


def GD(X,Y):
    loop_max = 10000                                        # 定义最大迭代次数
    sum1=0                                                  # 用于存储误差
    X_A = np.array(zeros((len(X), len(X[0]))))
    X_B = np.array(zeros((2, 1)))
    epsilon = 1e-20                                         # 定义最小误差
    alpha = 0.01                                            # 初始步长
    A = np.array([[float(0)], [0]])                         # 初始点为[0，0]
    N = len(X)                                              # 样本个数
    for Number in range(loop_max):
        if Number>5000:
             alpha=0.001                                    # 如果迭代次数大于5000，步长减小为0.001
        coefficient = np.dot(X,A)

        for i in range(len(Y)):
            coefficient[i] = coefficient[i]-Y[i]            # 此矩阵存储的为对应样本梯度中的系数部分
        for i in range(N):
            for j in range(2):
                X_A[i][j] = X[i][j]*coefficient[i][0]       # X_A矩阵用于存储各个样本的梯度
        for i in range(N):
            for j in range(2):
                X_B[j][0] +=X_A[i][j]                       # X_B矩阵用于存储将各个样本梯度求和的结果
        X_B[0][0] /= N
        X_B[1][0] /= N
        for i in range(2):
            A[i][0] =A[i][0]- X_B[i][0]*alpha               # 进行迭代
        for i in range(len(X_B)):
            sum1 += ((X_B[i][0])**2)**0.5
        if sum1 < epsilon:                                  # 梯度足够小时，终止迭代过程
            break
    return A


mean = np.array([2, 1])
cov = np.array([[1, 0.5], [0.5, 2]])
A = np.random.multivariate_normal(mean, cov, 30)
print(A)
x1 = []
y1 = []
x = np.array(ones((30, 2)))
y = np.array(ones((30, 1)))
x2 = np.arange(-5, 5, 0.1)
for i in range(30):
    x[i][1] = A[i][0]
    x1.append(A[i][0])
    y[i][0] = A[i][1]
    y1.append(A[i][1])
plt.scatter(x1, y1, color='red')
A = GD(x, y)
B = close_formed(x, y)
y2 = A[0][0]+A[1][0]*x2
plt.plot(x2, y2, color='blue')
print(A)
print(B)
plt.show()
