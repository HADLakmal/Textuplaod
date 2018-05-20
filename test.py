import matplotlib.pyplot as plt

import networkx as nx
import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from handler import Handler
import collections
import queue



G = nx.read_graphml("CombinedMinimizedGraph.graphml")
print(nx.to_numpy_matrix(G))
