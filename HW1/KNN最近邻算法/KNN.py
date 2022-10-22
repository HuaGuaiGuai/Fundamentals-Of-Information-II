import numpy as np
from sklearn import datasets
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.model_selection as sk_model


def DataSet():
    """ 数据集处理函数
        1.读取数据
        2.按照一定比例将数据集划分为训练集和验证集
        3.将对应数据和Label转化为np.array格式
    """
    iris = datasets.load_iris()
    # 划分数据集，验证集占20%
    x_tr, x_te, y_tr, y_te = sk_model.train_test_split(iris.data, iris.target, test_size=0.2)
    x_tr = np.array(x_tr)  # 转化为array格式
    x_te = np.array(x_te)
    y_tr = np.array(y_tr)
    y_te = np.array(y_te)
    return x_tr, x_te, y_tr, y_te


def View():
    """
        数据可视化函数
    """
    iris = datasets.load_iris()
    # 将数据转化为DF形式用于绘图
    iris_df = pd.DataFrame(iris['data'], columns=['sepal length(cm)',
                                                  'sepal width(cm)',
                                                  'petal length(cm)',
                                                  'petal width(cm)'])
    iris_df['Species'] = iris.target
    sns.relplot(x='petal length(cm)', y='petal width(cm)', data=iris_df, hue='Species')
    plt.show()  # 绘制在二维平面内的六张图片


def KNN(x_predict, x, y, k):
    """
        KNN函数
        1.x_predict代表用于分类的输入值，x代表训练集输入，label代表训练集标签，k代表k最近邻中的参数K
        2.x_size为训练集的行数,A,B,C为中间变量，无重大意义
        3.本函数中距离采用欧式距离
    """
    a1 = 0
    c = 0
    x_size = x.shape[0]  # 求取数据行数，也就是求取训练集的个数
    distance = (np.tile(x_predict, (x_size, 1)) - x) ** 2  # tile函数的作用是将待预测输入列表进行纵向复制，从而求取距离平方
    sum_distance = distance.sum(axis=1)  # 对获得的距离平方进行求和
    sqrt_distance = sum_distance ** 0.5  # 开方获得欧氏距离（这一步可省略）
    order = sqrt_distance.argsort()  # 随后对欧氏距离进行排序，按从小到大的顺序排序，而返回值为排序前的index
    neighbor_set = {}  # 建立一个空字典，后续使用
    for j in range(k):
        neighbor_point = y[order[j]]  # 统计前k个点中，不同种类的点的个数
        neighbor_set[neighbor_point] = neighbor_set.get(neighbor_point, 0) + 1
    for j in range(len(neighbor_set.keys())):  # 根据数量最多的种类进行判决
        b = list(neighbor_set.keys())
        if neighbor_set[b[j]] > a1:
            a1 = neighbor_set[b[j]]
            c = b[j]
    return c


right_number_train = 0
right_number_test = 0
x_train, x_test, y_train, y_test = DataSet()
number_test = x_test.shape[0]
number_train = x_train.shape[0]
label = []
print(x_test[0])
for i in range(number_test):  # 计算验证集的分类正确率
    a = KNN(x_test[i], x_train, y_train, 1)
    if a == y_test[i]:
        right_number_test = right_number_test + 1
print('验证集分类正确率为：', right_number_test / number_test)

for i in range(number_train):  # 计算训练集的正确率
    a = KNN(x_train[i], x_train, y_train, 1)
    if a == y_train[i]:
        right_number_train = right_number_train + 1
print(('训练集分类正确率为：', right_number_train / number_train))
rate = (right_number_test + right_number_train) / (number_test + number_train)
print('分类正确率为:', rate)
