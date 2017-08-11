
# coding: utf-8

# ### Machine Learning Application - MNIST Classification
# UDSL-SNU Big Data Academy   
# 20170725

# ##### Import libraries

# In[5]:

import h5py 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from sklearn import svm
from sklearn import neural_network


# ##### Load dataset

# In[2]:

with h5py.File('data/MNIST_vector.h', 'r') as f:
    X = np.asarray(f['X'])
    y = np.asarray(f['y'])


# ##### Preprocessing
# Intensiti normalization ( 0 to 1 )  
# casting int to float

# In[3]:

X = X / 255. 


# Train set, Test set split

# In[6]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y)#stratify == train 과 test 의 distribution 


# In[7]:

print ('Total {} Images'.format(X.shape[0]))
print ('Splitted Train({}) and Test({})'.format(X_train.shape[0], X_test.shape[0]))


# Label distribution

# In[8]:

hist_train = plt.hist(y_train, range=[0,10])
hist_test = plt.hist(y_test, range=[0,10])

print('Train distribution')
print(hist_train[0] / np.sum(hist_train[0]))
print('Test distribution')
print(hist_test[0] / np.sum(hist_test[0]))


# ##### Classifier - Linear Kernel SVM
# fit model with training data

# In[9]:

linearSVM = svm.LinearSVC(C=.01, max_iter = 100)
linearSVM.fit(X_train, y_train)


# predicted value from SVM model

# In[10]:

y_pred_SVM_train = linearSVM.predict(X_train)
y_pred_SVM = linearSVM.predict(X_test)


# In[11]:

print('(SVM)Trainig accuracy : {}'.format(accuracy_score(y_train, y_pred_SVM_train)))
print('(SVM)Test accuracy : {}'.format(accuracy_score(y_test, y_pred_SVM)))
print('(SVM)Confusion Matrix :')
print(confusion_matrix(y_test, y_pred_SVM))


# ##### Classifier - Multi Layer Perceptron
# fit model with training dat
# 

# In[13]:

NN = neural_network.MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4, 
                                    solver='sgd', verbose=10, learning_rate_init=.1)
NN.fit(X_train, y_train)


# predicted value from MLP model

# In[14]:

y_pred_NN_train = NN.predict(X_train)
y_pred_NN = NN.predict(X_test)


# In[15]:

print('(NN)Trainig accuracy : {}'.format(accuracy_score(y_train, y_pred_NN_train)))
print('(NN)Test accuracy : {}'.format(accuracy_score(y_test, y_pred_NN)))
print('(NN)Confusion Matrix :')
print(confusion_matrix(y_test, y_pred_NN))


# ### Effect of image Normalization
# Normalize individual images to have the same mean and std

# In[16]:

X_n_train = np.zeros(X_train.shape)
X_n_test = np.zeros(X_test.shape)

for n in range(X_train.shape[0]) : 
    X_n_train[n] = ( X_train[n] - np.mean(X_train[n]) ) / np.std(X_train[n])
for n in range(X_test.shape[0]) : 
    X_n_test[n] = ( X_test[n] - np.mean(X_test[n]) ) / np.std(X_test[n])


# ##### SVM

# In[17]:

linearSVM = svm.LinearSVC(C=.01, max_iter = 100)
linearSVM.fit(X_n_train, y_train)


# predicted value from SVM model

# In[18]:

y_pred_SVM_train = linearSVM.predict(X_n_train)
y_pred_SVM_n = linearSVM.predict(X_n_test)


# In[19]:

print('(SVM)Trainig accuracy : {}'.format(accuracy_score(y_train, y_pred_SVM_train)))
print('(SVM)Test accuracy : {}'.format(accuracy_score(y_test, y_pred_SVM_n)))
print('(SVM)Confusion Matrix :')
print(confusion_matrix(y_test, y_pred_SVM_n))


# ##### MLP

# In[20]:

NN = neural_network.MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4, 
                                    solver='sgd', verbose=10, learning_rate_init=.1)
NN.fit(X_n_train, y_train)


# predicted value from MLP model

# In[21]:

y_pred_NN_train = NN.predict(X_n_train)
y_pred_NN_n = NN.predict(X_n_test)


# In[22]:

print('(NN)Trainig accuracy : {}'.format(accuracy_score(y_train, y_pred_NN_train)))
print('(NN)Test accuracy : {}'.format(accuracy_score(y_test, y_pred_NN_n)))
print('(NN)Confusion Matrix :')
print(confusion_matrix(y_test, y_pred_NN_n))


# ### Effect of image Standardization
# Standardize individual pixels to have the same mean and std
# We used the StandardScaler implemented in Sklearn

# In[23]:

ss = StandardScaler()
ss.fit(X_n_train)
X_st_train = ss.transform(X_n_train)
X_st_test = ss.transform(X_n_test)


# ##### SVM

# In[24]:

linearSVM = svm.LinearSVC(C=.01, max_iter = 100)
linearSVM.fit(X_st_train, y_train)


# predicted value from SVM model

# In[25]:

y_pred_SVM_train = linearSVM.predict(X_st_train)
y_pred_SVM_st = linearSVM.predict(X_st_test)


# In[26]:

print('(SVM)Trainig accuracy : {}'.format(accuracy_score(y_train, y_pred_SVM_train)))
print('(SVM)Test accuracy : {}'.format(accuracy_score(y_test, y_pred_SVM_st)))
print('(SVM)Confusion Matrix :')
print(confusion_matrix(y_test, y_pred_SVM_st))


# ##### MLP

# In[27]:

NN = neural_network.MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4, 
                                    solver='sgd', verbose=10, learning_rate_init=.1)
NN.fit(X_st_train, y_train)


# predicted value from MLP model

# In[ ]:

y_pred_NN_train = NN.predict(X_st_train)
y_pred_NN_st = NN.predict(X_st_test)


# In[ ]:

print('(NN)Trainig accuracy : {}'.format(accuracy_score(y_train, y_pred_NN_train)))
print('(NN)Test accuracy : {}'.format(accuracy_score(y_test, y_pred_NN_st)))
print('(NN)Confusion Matrix :')
print(confusion_matrix(y_test, y_pred_NN_st))


# ### Results

# In[ ]:

print('(SVM)Test accuracy : {}'.format(accuracy_score(y_test, y_pred_SVM)))
print('(NN)Test accuracy : {}'.format(accuracy_score(y_test, y_pred_NN)))
print('Normalization')
print('(SVM)Test accuracy : {}'.format(accuracy_score(y_test, y_pred_SVM_n)))
print('(NN)Test accuracy : {}'.format(accuracy_score(y_test, y_pred_NN_n)))
print('Standardization')
print('(SVM)Test accuracy : {}'.format(accuracy_score(y_test, y_pred_SVM_st)))
print('(NN)Test accuracy : {}'.format(accuracy_score(y_test, y_pred_NN_st)))


# In[ ]:



