'''
Created on Feb 16, 2011
k Means Clustering for Ch10 of Machine Learning in Action
@author: Peter Harrington
@author: Yi Lin, add visualization and change code to python3
'''
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))#map all elements to float()
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)

def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))#create centroid mat
    for j in range(n):#create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:,j])
        maxJ = max(dataSet[:,j])
        rangeJ = float(maxJ - minJ)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids
    
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))#create mat to assign data points 
                                      #to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):#for each data point assign it to the closest centroid
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        print(centroids)
        for cent in range(k):#recalculate centroids
            #B = clusterAssment[:,0].A
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster
            centroids[cent,:] = mean(ptsInClust, axis=0) #assign centroid to mean 
    return centroids, clusterAssment

if __name__ == "__main__":
    datMat = mat(loadDataSet('testSet.txt'))  # type: matrix
    myCentroids, clustAssing = kMeans(datMat, 4)

    # visualize the result
    p1 = datMat[nonzero(clustAssing[:, 0].A == 0)[0]]
    p2 = datMat[nonzero(clustAssing[:, 0].A == 1)[0]]
    p3 = datMat[nonzero(clustAssing[:, 0].A == 2)[0]]
    p4 = datMat[nonzero(clustAssing[:, 0].A == 3)[0]]

    plt.scatter(array(p1[:, 0]), array(p1[:, 1]), marker='o', c='y')
    plt.scatter(array(p2[:, 0]), array(p2[:, 1]), marker='o', c='r')
    plt.scatter(array(p3[:, 0]), array(p3[:, 1]), marker='o', c='g')
    plt.scatter(array(p4[:, 0]), array(p4[:, 1]), marker='o', c='c')

    plt.show()


