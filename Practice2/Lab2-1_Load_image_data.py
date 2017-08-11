
# coding: utf-8

# ### Machine Learning Application - Load Images
# UDSL-SNU Big Data Academy   
# 20170725

# ##### Import libraries

# In[1]:

import h5py 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer


# ##### Load Dataset

# In[2]:

with h5py.File('data/MNIST.h', 'r') as f:#h5py - 파일하나에 여러 변수를 저장할 수 있다.
    X = np.asarray(f['X'])
    y = np.asarray(f['y'])


# ##### Data observation

# In[3]:

print ('X.shape : {}'.format(X.shape))
print ('y.shape : {}'.format(y.shape))


# In[4]:

print ('X : ')
print (X)


# In[5]:

print ('X[0] :')
print (X[0])


# In[6]:

print ('y : ')
print (y)
print ('unique y :')
print (np.unique(y))


# ##### Plot image (X)
# 

# In[7]:

index_plot = 5231

plt.figure()
plt.imshow(X[index_plot], cmap='gray')
plt.axis('off')
plt.title('IMAGE label={}'.format(y[index_plot]))
plt.show()


# ##### Plot label distribution (y)

# In[8]:

plt.figure()
res_hist = plt.hist(y, range=[0,10])
plt.xlabel('y value')
plt.ylabel('count')
plt.title('Class distribution of MNIST')
plt.show()


# ##### histrogram details

# In[9]:

print('Centers')
print(res_hist[1])
print('Counts')
print(res_hist[0])


# ##### Label Binarize
# make numerical varible [3] to dummy varible [0 0 0 1 0 0 0 0 0 0]

# In[10]:

lb = LabelBinarizer()
lb.fit(y)
y_dummy = lb.transform(y)


# ##### Label Distribution

# In[11]:

print('np.sum(y_dummy, axis=0)')
print(np.sum(y_dummy, axis=0))
print('np.mean(y_dummy, axis=0)')
print(np.mean(y_dummy, axis=0))


# ##### Image Vectorization

# In[12]:

X_vec = X.reshape( (X.shape[0], X.shape[1] * X.shape[2]) )
print('Vectorized image shape')
print(X_vec.shape)


# ##### Save vectorized image for classification

# In[13]:

with h5py.File('data/MNIST_vector.h', 'w') as f:
    f.create_dataset('X', data = X_vec)
    f.create_dataset('y', data = y)


# In[ ]:



