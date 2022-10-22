import numpy as np
import matplotlib.pyplot as plt

x01 = []
x02 = []
x11 = []
x12 = []                        # 定义一些空列表，用于后续绘图使用
# 利用np中的函数生成50个高斯随机点，用于后续验证结果
x = np.random.multivariate_normal(np.array([0, 0]), np.array([[100, 99], [99, 100]]), 50)
# 利用Gibbs Sampling产生50个高斯随机点
for j in range(50):
    # 由于迭代过程只需要保持一个特征不变，来确定另一个参数，也就是说初始点只用随机选取一个特征即可
    x1 = 0
    # x2为初始点随机选取的值
    x2 = np.random.randint(-20, 20)
    # 每个点迭代100次
    for i in range(100):
        # 保证x2不变，确定x1
        x1 = np.random.normal(0.99*x2, 100-99**2/100)
        # 保证x1不变，确定x2
        x2 = np.random.normal(0.99*x1, 100-99**2/100)
    #将迭代完成的点添加到绘图列表中
    x01.append(x1)
    x02.append(x2)
# 将利用高斯函数产生的点添加到另一个绘图列表中
for i in range(len(x)):
    x11.append(x[i][0])
    x12.append(x[i][1])
# 绘图，进行对比。
plt.subplot(121)
plt.scatter(x01, x02)
plt.title("Gibbs")
plt.subplot(122)
plt.scatter(x11, x12, color="red")
plt.title("origin")
plt.show()
