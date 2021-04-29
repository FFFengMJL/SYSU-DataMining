import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import random


def getData(filename):
    '''读取数据

    Args:
      filename 文件名（包含路径）

    Returns:
      data
    '''
    np.set_printoptions(suppress=True)

    file = open(filename)
    lines = file.readlines()
    lineNum = len(lines)

    data = np.zeros((lineNum, 3), dtype=np.longcomplex)
    index = 0

    for line in lines:
        line = line.strip()
        lineData = line.split(' ')
        data[index, :] = lineData[0: 3]
        index += 1

    return data


def gradientDecent(learning_rate=0.00015, theta_0=np.longcomplex(0.0), theta_1=np.longcomplex(0.0), theta_2=np.longcomplex(0.0), iteration_time=1500000, show_margin=100000, random_points=0):
    '''梯度下降训练线性模型

    Args:
        learning_rate: 学习率
        theta_0: 初始参数0
        theta_1: 初始参数1
        theta_2: 初始参数2
        iteration_time: 迭代次数
        show_margin: 数据展示的迭代次数间隔

    Returns:
        iterationTimes: 迭代次数数组
        trainingErrors: 训练误差
        testingErrors: 测试误差
        resTable: 结果表
    '''
    # 获取数据
    trainSet = getData('../data/dataForTrainingLinear.txt')  # 训练集
    testSet = getData('../data/dataForTestingLinear.txt')  # 测试集
    trainSetSize = len(trainSet)  # 训练集大小
    testSetSize = len(testSet)  # 测试集大小

    resTable = PrettyTable(["iteration_times", "theta_0", "theta_1",
                            "theta_2", "training_error", "testing_error"])  # 用于美观显示
    trainingErrors = []  # 训练误差
    testingErrors = []  # 测试误差
    iterationTimes = []  # 迭代次数

    for iterTime in range(1, iteration_time + 1):
        sum_0 = np.longcomplex(0)
        sum_1 = np.longcomplex(0)
        sum_2 = np.longcomplex(0)

        if (random_points == 0):
            # 正常梯度下降迭代
            for trainIndex in range(trainSetSize):
                # 初始化三个参数的和，用于更新 theta
                h_x = theta_0 + theta_1 * \
                    trainSet[trainIndex][0] + theta_2 * \
                    trainSet[trainIndex][1]  # 计算结果
                sum_0 += (h_x - trainSet[trainIndex][2])
                sum_1 += (h_x - trainSet[trainIndex]
                          [2]) * trainSet[trainIndex][0]
                sum_2 += (h_x - trainSet[trainIndex]
                          [2]) * trainSet[trainIndex][1]
        else:
            # 随机梯度下降迭代
            for _ in range(random_points):
                index = random.randint(0, trainSetSize - 1)
                h_x = theta_0 + theta_1 * \
                    trainSet[index][0] + theta_2 * \
                    trainSet[index][1]  # 计算结果
                sum_0 += (h_x - trainSet[index][2])
                sum_1 += (h_x - trainSet[index]
                          [2]) * trainSet[index][0]
                sum_2 += (h_x - trainSet[index]
                          [2]) * trainSet[index][1]

        # 迭代参数
        theta_0 = theta_0 - learning_rate * \
            (sum_0 / (trainSetSize if (random_points == 0) else random_points))
        theta_1 = theta_1 - learning_rate * \
            (sum_1 / (trainSetSize if (random_points == 0) else random_points))
        theta_2 = theta_2 - learning_rate * \
            (sum_2 / (trainSetSize if (random_points == 0) else random_points))

        # 每间隔 show_margin 次进行一次结果展示
        if (iterTime % show_margin == 0):
            print("迭代次数: ", iterTime)
            iterationTimes.append(iterTime + 1)
            trainVar = np.longcomplex(0.0)  # 训练集误差
            testVar = np.longcomplex(0.0)  # 测试集误差

            for trainSetIndex in range(trainSetSize):
                H_x = np.longcomplex(theta_0 + theta_1 *
                                     trainSet[trainSetIndex][0] + theta_2 *
                                     trainSet[trainSetIndex][1])  # 计算结果
                # (y* - y_i)^2 用于计算方差
                trainVar += np.square(np.longcomplex(H_x) -
                                      np.longcomplex(trainSet[trainSetIndex][2]))

            trainError = np.longcomplex(trainVar * 1.0 / trainSetSize)
            trainingErrors.append(trainError)

            for testSetIndex in range(testSetSize):
                H_x = np.longcomplex(theta_0 + theta_1 *
                                     testSet[testSetIndex][0] + theta_2 *
                                     testSet[testSetIndex][1])   # 计算结果
                testVar += np.square(np.longcomplex(H_x) -
                                     np.longcomplex(testSet[testSetIndex][2]))

            testError = np.longcomplex(testVar * 1.0 / testSetSize)
            testingErrors.append(testError)

            resTable.add_row([iterTime, theta_0, theta_1,
                              theta_2, trainError, testError])

    # print(resTable)

    return (iterationTimes, trainingErrors, testingErrors, resTable)


(iterationTimes, trainingErrors, testingErrors,
 resTable) = gradientDecent(learning_rate=0.0002, iteration_time=2000, show_margin=200, random_points=0)


print(resTable)

# 画图
plt.figure()
plt.plot(iterationTimes, trainingErrors, "+-",
         c="r", linewidth=1, label="training error")
plt.plot(iterationTimes, testingErrors, "X-",
         c="b", linewidth=1, label="testing error")

plt.xlabel("Iteration Times")
plt.ylabel("Error")
plt.legend()
plt.title("Gradient Descent")
plt.show()
