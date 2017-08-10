
# coding: utf-8

# In[11]:

import numpy as np
import os
import matplotlib
matplotlib.use('Qt5Agg')  # You can delete this if it occurs problems
import matplotlib.pyplot as plt
import scipy
import time


# In[12]:

# Data generation
np.random.seed(1234)
mean1 = [1, 6]
cov1 = [[6, 0], [2, 8]]
x1, y1 = np.random.multivariate_normal(mean1,cov1,300).T


# In[13]:

np.random.seed(1234)
mean2 = [-1, -6]
cov2 = [[5, 2], [1.3, 4]]
x2, y2 = np.random.multivariate_normal(mean2,cov2,300).T


# In[14]:

np.random.seed(1234)
mean3 = [6, -2]
cov3 = [[3, 0], [0, 7]]
x3, y3 = np.random.multivariate_normal(mean3,cov3,300).T


# In[15]:

x = np.hstack((x1,x2,x3))
y = np.hstack((y1,y2,y3))
data = np.vstack((x,y))


# In[16]:

# Plot : labeled data points
plt.ion()
plt.figure()
plt.plot(x1,y1,'bo')
plt.plot(x2,y2,'ro')
plt.plot(x3,y3,'go')
plt.title('Ground truth')
plt.ylim(-15,15)
plt.xlim(-8,12)
plt.pause(0.5)
plt.show()
input('Press <Enter> to continue')  # You can delete this if it occurs problems


# In[17]:

# Plot : unlabeled data points
plt.figure()
plt.plot(x,y,'bo')
plt.title('Data to be clustered')
plt.ylim(-15,15)
plt.xlim(-8,12)
plt.pause(0.5)
plt.show()
input('Press <Enter> to continue')  # You can delete this if it occurs problems


# In[18]:

# K means clustering algorithm
nCluster = 3
np.random.seed(8888)
centerIdx = np.random.permutation(len(x))[0:nCluster]
centerX = x[centerIdx[0:nCluster]]
centerY = y[centerIdx[0:nCluster]]
dist = np.zeros((x.shape[0],nCluster))
plt.figure()


# In[19]:

for i in range(0,10):
    # Calculate distance from the centers to the data points
    for j in range(0,nCluster):
        dist[:,j] = np.sqrt(np.square(x-centerX[j])+np.square(y-centerY[j]))

    # A point belongs to the cluster of which center is the closest from here
    clusterIdx = np.argmin(dist, axis=1)

    # Update center point
    for j in range(0,nCluster):
        centerX[j] = np.mean(x[np.where(clusterIdx==j)])
        centerY[j] = np.mean(y[np.where(clusterIdx==j)])

    # Plot for real-time visualization
    plt.clf()
    plt.plot(x[np.where(clusterIdx==0)],y[np.where(clusterIdx==0)],'bo')
    plt.plot(x[np.where(clusterIdx==1)],y[np.where(clusterIdx==1)],'ro')
    plt.plot(x[np.where(clusterIdx==2)],y[np.where(clusterIdx==2)],'go')
    plt.plot(centerX,centerY,'kX', markerSize=20)
    plt.title('K means clustering (iteration='+str(i+1)+')')
    plt.ylim(-15,15)
    plt.xlim(-8,12)
    plt.show()
    plt.pause(0.5)


# In[ ]:



