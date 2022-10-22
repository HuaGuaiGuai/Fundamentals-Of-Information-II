import matplotlib.pyplot as plt
from numpy import *
import numpy as np


def Mini(x, centers):
    """
    : param centers:
    : param x: 一个样本点
    : return: 最小距离
    """
    min_dist = 2000000000000000.0
    m = np.shape(centers)[0]  # 当前已经初始化聚类中心的个数
    n = len(x)
    for i in range(m):
        # 计算该样本与每个聚类中心之间的距离
        d = 0
        for j in range(n):
            d += (x[j] - centers[i, j]) ** 2
        # 选择最短距离
        if min_dist > d:
            min_dist = d
    return min_dist


def Initialization_a2(x, k):
    """
    :param x:样本
    :param k:k——means参数
    :return:各类别中心点
    """
    m, n = np.shape(x)
    centers = np.mat(np.zeros((k, n)))
    # 随机选取一个样本点作为第一个聚类中心
    index = np.random.randint(0, m)
    centers[0, ] = np.copy(x[index, ])

    distance = [0.0 for _ in range(m)]  # 用于存储距离
    for i in range(1, k):
        sum_dis = 0
        for j in range(m):
            distance[j] = Mini(x[j, ], centers[0:i], )
            sum_dis += distance[j]  # 将所有最小距离求和

        sum_dis *= np.random.random()  # 在距离和的范围内随机选取一个数字
        for j, dj in enumerate(distance):
            sum_dis -= dj
            if sum_dis < 0:  # 持续作差直至找到使得作差结果为负数
                centers[i] = np.copy(x[j, ])
                break
    return centers


def Initialization_a0(x, k):
    index = []
    m, n = np.shape(x)
    centers = np.mat(np.zeros((k, n)))  # 建立中心点向量矩阵
    for i in range(k):  # 随机产生k个随机数
        b = np.random.randint(0, m)
        if b not in index:  # 如果产生的随机数在已有列表中不存在，则直接选为中心点
            index.append(b)
            centers[i, ] = np.copy(x[b, ])
        else:
            for j in range(100):  # 如果产生的随机数在列表中已经存在，则需要重新产生随机数
                b = np.random.randint(0, m)
                if b not in index:
                    index.append(b)
                    centers[i, ] = np.copy(x[b, ])
                    break
    return centers


def Initialization_a00(x, k):
    m, n = np.shape(x)
    centers = np.mat(np.zeros((k, n)))
    # 随机选取一个样本点作为第一个聚类中心
    index = np.random.randint(0, m)
    centers[0, ] = np.copy(x[index, ])
    distance = [0.0 for _ in range(m)]
    for i in range(1, k):  # 针对第i个中心点
        for j in range(m):  # 针对第j个样本
            distance[j] = Mini(x[j, ], centers[0:i], )  # 求取每个样本到达所有已经初始化中心点的最小距离
        max_d = max(distance)  # 选取最小距离最大的样本点作为下一个中心点
        for j in range(m):
            if distance[j] == max_d:
                centers[i, ] = np.copy(x[j, ])
    return centers


def Gauss_random(mean1, cov1, mean2, cov2, mean3, cov3, num):
    x1 = np.random.multivariate_normal(mean1, cov1, num)
    x2 = np.random.multivariate_normal(mean2, cov2, num)
    x3 = np.random.multivariate_normal(mean3, cov3, num)
    y1 = []
    for i in range(3):
        for j in range(num):
            y1.append(i)
    x4 = np.vstack((x1, x2))
    x = np.vstack((x3, x4))
    return x, y1


def Z_optimize(x, z, centers):
    """
    :param x: 样本集合
    :param z: 对应点归类
    :param centers: 各类中心点坐标
    :return: 更新后的归类
    """
    ind = 0
    for j in range(len(x)):
        dj = [0 for _ in range(len(centers))]  # 用于存储样本点到每个中心点的距离
        for i in range(len(centers)):  # 循环求取样本到每个点的距离
            d = 0
            ind = 0
            for b in range(len(x[0])):
                d += (x[j, b] - centers[i, b]) ** 2
            dj[i] = d
        min_dis = min(dj)  # 将距离最近的聚类中心点更新为样本的分类
        for c in range(len(dj)):
            if min_dis == dj[c]:
                ind = c
        z[j] = ind
    return z


def C_optimize(x, z, centres):
    """
    :param x: 样本集合
    :param z: 所述分类中心标签
    :param centres: 分类中心点
    :return: 更新后的分类中心点
    """
    num = [0 for _ in range(len(centres))]  # 用于存储每个类别下的数据点的个数
    d = np.mat(np.zeros((len(centres), len(x[0]))))  # 用于存储不同种类下各个数据点求和
    for i in range(len(centres)):
        for j in range(len(z)):
            if z[j] == i:
                d[i] = d[i] + x[j]
                num[i] += 1
    for i in range(len(centres)):
        d[i] = d[i] / num[i]  # 不同种类下的数据点求和除以对应的数据点个数
    return d


def K_means(x, centers):
    z = [0 for _ in range(len(x))]
    n = 1000
    for i in range(n):
        z = Z_optimize(x, z, centers)
        centers = C_optimize(x, z, centers)
    return z, centers


def Error(z, centers, x):
    error = 0
    for i in range(len(x)):
        for j in range(len(x[0])):
            error += (x[i, j] - centers[z[i], j]) ** 2
    return error


def Choose_k(k, error):
    maximize = -2
    k_final = 0
    for i in range(1, len(k) - 1):
        k1 = [-1, error[i - 1] - error[i]]
        k2 = [1, error[i + 1] - error[i]]
        print(k1, k2)
        k1[0] /= (k1[0] ** 2 + k1[1] ** 2) ** 0.5
        k1[1] /= (k1[0] ** 2 + k1[1] ** 2) ** 0.5
        k2[0] /= (k2[0] ** 2 + k2[1] ** 2) ** 0.5
        k2[1] /= (k2[0] ** 2 + k2[1] ** 2) ** 0.5
        middle = k1[0] * k2[0] + k1[1] * k2[1]
        print(middle)
        if middle > maximize:
            maximize = middle
            k_final = i + 1
    return k_final


m1 = np.array([0, 0])
m2 = np.array([2, 0])
m3 = np.array([0, 2])
c1 = np.array([[1, 0], [0, 1]])
c2 = np.array([[0.5, 0], [0, 1]])
c3 = np.array([[1, 0.3], [0.3, 1]])
X, y = Gauss_random(m1, c1, m2, c2, m3, c3, 25)
centers0 = Initialization_a0(X, 3)
centers1 = Initialization_a2(X, 3)
centers2 = Initialization_a00(X, 3)
z0, centers0 = K_means(X, centers0)
z1, centers1 = K_means(X, centers1)
z2, centers2 = K_means(X, centers2)

# k = []
# E = []
# for i in range(1,10):
#     centers1 = Initialization_a2(x, i)
#     z1, centers1 = K_means(x,centers1)
#     k.append(i)
#     E.append(Error(z1, centers1, x))
# print(Choose_k(k, E))
# plt.plot(k, E, marker='o', mec='r', mfc='w')
# plt.show()
# ----------------------------------------------------原始绘图-----------------------------------------------------------------------
x001 = []
x002 = []
x011 = []
x012 = []
x021 = []
x022 = []
for a in range(len(z0)):
    if y[a] == 0:
        x001.append(X[a, 0])
        x002.append(X[a, 1])
    elif y[a] == 1:
        x011.append(X[a, 0])
        x012.append(X[a, 1])
    else:
        x021.append(X[a, 0])
        x022.append(X[a, 1])
plt.subplot(231)
plt.scatter(x001, x002, color='blue')
plt.scatter(x011, x012, color='green')
plt.scatter(x021, x022, color='yellow')
plt.title('origin')
plt.subplot(232)
plt.scatter(x001, x002, color='blue')
plt.scatter(x011, x012, color='green')
plt.scatter(x021, x022, color='yellow')
plt.title('origin')
plt.subplot(233)
plt.scatter(x001, x002, color='blue')
plt.scatter(x011, x012, color='green')
plt.scatter(x021, x022, color='yellow')
plt.title('origin')
# ----------------------------------------------------分类后绘图----------------------------------------------------------------------
x01 = []
x02 = []
x11 = []
x12 = []
x21 = []
x22 = []
for a in range(len(z0)):
    if z1[a] == 0:
        x01.append(X[a, 0])
        x02.append(X[a, 1])
    elif z1[a] == 1:
        x11.append(X[a, 0])
        x12.append(X[a, 1])
    else:
        x21.append(X[a, 0])
        x22.append(X[a, 1])
plt.subplot(234)
plt.scatter(x01, x02, color='blue')
plt.scatter(x11, x12, color='green')
plt.scatter(x21, x22, color='yellow')
# plt.scatter(x[:,0],x[:,1])
a01 = []
a02 = []
for a in range(len(centers1)):
    a01.append(centers1[a, 0])
    a02.append(centers1[a, 1])
plt.scatter(a01, a02, color='red')
plt.title('k-means++')

# -------------------------------------------------------------------------------------------------------------------------------------
x01 = []
x02 = []
x11 = []
x12 = []
x21 = []
x22 = []
for a in range(len(z0)):
    if z0[a] == 0:
        x01.append(X[a, 0])
        x02.append(X[a, 1])
    elif z0[a] == 1:
        x11.append(X[a, 0])
        x12.append(X[a, 1])
    else:
        x21.append(X[a, 0])
        x22.append(X[a, 1])
plt.subplot(235)
plt.scatter(x01, x02, color='blue')
plt.scatter(x11, x12, color='green')
plt.scatter(x21, x22, color='yellow')
# plt.scatter(x[:,0],x[:,1])
a01 = []
a02 = []
for a in range(len(centers0)):
    a01.append(centers0[a, 0])
    a02.append(centers0[a, 1])
plt.scatter(a01, a02, color='red')
plt.title('random')
# --------------------------------------------------------------------------------------------------------------------------------
x01 = []
x02 = []
x11 = []
x12 = []
x21 = []
x22 = []
for a in range(len(z0)):
    if z2[a] == 0:
        x01.append(X[a, 0])
        x02.append(X[a, 1])
    elif z2[a] == 1:
        x11.append(X[a, 0])
        x12.append(X[a, 1])
    else:
        x21.append(X[a, 0])
        x22.append(X[a, 1])
plt.subplot(236)
plt.scatter(x01, x02, color='blue')
plt.scatter(x11, x12, color='green')
plt.scatter(x21, x22, color='yellow')
# plt.scatter(x[:,0],x[:,1])
a01 = []
a02 = []
for a in range(len(centers2)):
    a01.append(centers2[a, 0])
    a02.append(centers2[a, 1])
plt.scatter(a01, a02, color='red')
plt.title('far')
plt.show()
