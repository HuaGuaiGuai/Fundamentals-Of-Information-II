import numpy as np
from numpy import *
from sklearn import datasets
import heapq
import matplotlib.pyplot as plt


def DataSet():
    iris = datasets.load_iris()                         # 导入鸢尾花数据集
    data = iris.data
    label = iris.target
    data = np.array(data)                               # 转为为numpy格式
    data = data.T                                       # 对特征向量进行转置
    label = np.array(label)

    return data, label                                  # 输出特征矩阵及其标签值


def Se_data():
    iris = datasets.load_iris()  # 导入鸢尾花数据集
    da = iris.data
    label = iris.target
    data = []
    for num in range(len(label)):
        if label[num] == 0:
            data.append(list(da[num]))
    data = np.array(data)
    data = data.T
    return data


def Data_transform(data):
    m = len(data[0])                                    # 定义样本数量
    n = len(data)                                       # 定义样本维度
    b = np.sum(data, axis=1, keepdims=True)             # 对m个样本求和
    b /= m                                              # 求取每个特征的均值
    for i in range(n):
        for j in range(m):
            data[i][j] -= b[i][0]                       # 将每个样本进行平移
    # a = data ** 2
    # sigma = np.sum(a, axis=1, keepdims=True)
    # sigma /= m                                          # 求取方差
    # sigma = sigma ** 0.5                                # 求取每个样本的标准差
    # for i in range(n):
    #     for j in range(m):
    #         data[i][j] /= sigma[i][0]                   # 完成数据标准化工作

    return data                                         # 返回标准化数据


def Cov_matrix(data):
    m = len(data[0])                                    # m为样本数量
    cov_matrix = np.dot(data, data.T)                   # 求取协方差矩阵
    cov_matrix /= m
    return cov_matrix                                   # 协方差矩阵


def Feature(cov_matrix):
    k = 2                                               # 设定k值
    u, s, v = np.linalg.svd(cov_matrix)                 # 奇异值分解
    u_reduce = u.T[:k]                                  # 求取转换矩阵
    return u_reduce, s, k


def Select_k(s):
    s_lamda = 0                                         # 定义表达式分母
    n_lamda = 0                                         # 定义表达式分子
    a = 0.05
    k = 0
    for i in range(len(s)):
        s_lamda += s[i][i]                              # 求取所有特征值的和
    for i in range(len(s)):                             # 求取前k个特征值的和
        n_lamda += s[i][i]
        if n_lamda/s_lamda > 1 - a:
            k = i
            break
    return k


def PCA(data, u_reduce):
    z = np.dot(u_reduce, data)
    x_approx = np.dot(u_reduce.T, z)
    z *= -1
    return z, x_approx


x, y = DataSet()
x = Se_data()
x = Data_transform(x)
cov = Cov_matrix(x)
print(cov)
u, s, k = Feature(cov)
for i in range(len(u)):
    print('第', i+1, '主成分向量为', u[i])
Z, X_A = PCA(x, u)
E = x - X_A
E = E ** 2
E = np.sum(E, axis=1, keepdims=True)
E = np.sum(E, axis=0, keepdims=True)
E /= len(x)
print('重构误差为', E[0,0])
x01 = []
x02 = []
x11 = []
x12 = []
x21 = []
x22 = []
for i in range(len(Z[0])):
    if y[i] == 0:
        x01.append(Z[0][i])
        x02.append(Z[1][i])
    elif y[i] == 1:
        x11.append(Z[0][i])
        x12.append(Z[1][i])
    else:
        x21.append(Z[0][i])
        x22.append(Z[1][i])
plt.scatter(x01, x02, color='blue')
plt.scatter(x11, x12, color='red')
plt.scatter(x21, x22, color='green')
plt.show()
n1 = 0
s1 = 0
for i in range(len(s)):
    s1 += s[i] # 求取所有特征值的和
for i in range(k):  # 求取前k个特征值的和
    n1 += s[i]
print('特征保有率为', n1/s1)