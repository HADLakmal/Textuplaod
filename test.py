import numpy as np
partition = []
for i in range(0,10):
    temppart = []
    for k in range(0,10):
        temppart.append(k)
    partition.append(temppart)

np.savetxt('test.txt', partition,fmt='%r')
