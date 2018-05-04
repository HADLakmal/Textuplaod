import networkx as nx
import numpy as np


class Handler:

    #get exact node list from graph partition
    def indexArray(nodeList,k,lables):
        array = []
        for r in range(0,k):
            array.append([])
            count = 0
            for i in lables:
                count+=1
                graph_count = 0
                for a in nodeList:
                    graph_count+=1
                    if(r==i and count==graph_count):
                        array[r].append(a)
                        break
        return array
    def alphaCut(A):
        #define zero matrix
        degree = np.zeros((A.shape[0],A.shape[0]))
        #define ones matrix
        oneMatrix = np.ones((A.shape[0],A.shape[0]))
        #get sum of row
        rowsum = A.sum(axis=0)
        #create degree matrix
        for j in range(0, A.shape[0]):
            degree[j,j] = rowsum[0,j]
        #Get alpha cut matrix
        maltiply = np.matmul(oneMatrix.transpose(), degree)

        numerator = np.matmul(maltiply.transpose(),maltiply)
        denominator = np.matmul(maltiply,oneMatrix)
        value = np.subtract(numerator,denominator)
        return value-A

