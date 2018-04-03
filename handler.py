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
