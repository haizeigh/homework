import numpy as np
import matplotlib.pyplot as plt


def initCenter(dataSet, k):
    numSamples, dim = dataSet.shape
    center = np.zeros((k,dim))
    for i in range(k):
        index = int(np.random.uniform(0, numSamples))
        center[i,:] = dataSet[index,:]
    return center

# 慕课网上抄写的demo  kmeans聚类
def disToCenter(point, center):
    k = center.shape[0]
    dis = np.zeros(k)

    for i in range(k):
        dis[i] = np.sqrt(np.sum(np.power(point - center[i,:],2)))
    return dis


def kmeans(dataSet, k, iterNum):
    numSamples = dataSet.shape[0]
    iterCount = 0

    cluster = np.zeros(numSamples)
    clusterChange = True

    center = initCenter(dataSet,k)
    while clusterChange and iterCount<iterNum:
        iterCount += 1
        clusterChange = False
        # 分配点到cluster
        for i in range(numSamples):
            disArr = disToCenter(dataSet[i,:],center)
            minIndex = np.argmin(disArr)
            if cluster[i] != minIndex:
                clusterChange = True
                cluster[i] = minIndex

        # 更新center
        for j in range(k):
            newcluster = dataSet[np.nonzero(cluster[:] == j)[0]]
            center[j,:]=np.mean(newcluster, axis=0)
    print("结束")
    return center,cluster


def show(dataSet, k, center, cluster):
    numSamples, dim = dataSet.shape
    mark = ['or', 'ob', 'og', 'om']

    for i in range(numSamples):
        markIndex = int(cluster[i])
        plt.plot(dataSet[i,0], dataSet[i,1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dm']
    for i in range(k):
        plt.plot(center[i, 0], center[i, 1], mark[i], markersize = 17)


    plt.show()


def main():
    print("读取数据")
    file = open("/Users/xmly/desktop/testSet 2.txt")
    dataSet = []
    for line in file:
        lineArr = line.split("\t")
        dataSet.append([float(lineArr[0]),np.sqrt(float(lineArr[1]))])

    thelen = len(dataSet)
    mark = ['or', 'ob', 'og', 'om']
    print(thelen/2)
    for i in range(int(thelen/2)):
        plt.plot(dataSet[2*i][0], dataSet[2*i][1], mark[i])
        plt.plot(dataSet[2*i+1][0], dataSet[2*i+1][1], mark[i])
    plt.show()

    print("聚簇")
    dataSet = np.mat(dataSet)
    k = 3
    center,cluster = kmeans(dataSet,k,100)

    print("显示簇")
    show(dataSet, k, center, cluster)

main()