import matplotlib.pyplot as plt

import networkx as nx
import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

G=nx.Graph()

G.add_edge('a','b',weight=0.8)
G.add_edge('b','c',weight=0.9)
G.add_edge('a','c',weight=0.6)
"""
G.add_edge('a','d',weight=0.1)
G.add_edge('c','f',weight=0.2)
G.add_edge('f','d',weight=0.7)
G.add_edge('e','f',weight=0.8)
G.add_edge('e','d',weight=0.6)
"""
G.add_edge('a','d',weight=0)
G.add_edge('c','f',weight=0)
G.add_edge('f','d',weight=0.3)
G.add_edge('e','f',weight=0.3)
G.add_edge('e','d',weight=0.3)

A = nx.adjacency_matrix(G)
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

M = value-A



#eigen calculation

eigenvalues, eigenvectors = np.linalg.eig(M)

#define K
k=3
tempEigenValues = np.absolute(eigenvalues)
print(tempEigenValues)
idx = tempEigenValues.argsort()[:k][::]
print(idx)
print(eigenvectors)
eigenValues = tempEigenValues[idx]
eigenVectors = eigenvectors[:,idx]

print(...)
print("Second tuple of eig\n", eigenVectors)

z = eigenVectors
#normalize the matrix
for i in range(0,eigenVectors.shape[0]):
    total = 0
    for j in range(0,eigenVectors.shape[1]):
        total += eigenVectors.item((i,j))**2
    z[i]=+z[i]/(total**(1/2))
print(z)

kmeans = KMeans(n_clusters=k, random_state=0).fit(z)

print(kmeans.labels_)

'''
elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] >0.7]
esmall=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] <=0.7]

pos=nx.spring_layout(G) # positions for all nodes

# nodes
nx.draw_networkx_nodes(G,pos,node_size=700)

# edges
nx.draw_networkx_edges(G,pos,edgelist=elarge,
                    width=6)
nx.draw_networkx_edges(G,pos,edgelist=esmall,
                    width=6,alpha=0.5,edge_color='b',style='dashed')

# labels
nx.draw_networkx_labels(G,pos,font_size=20,font_family='sans-serif')

plt.axis('off')
plt.savefig("weighted_graph.png") # save as png
plt.show() # display
'''
