
# coding: utf-8

# ### Machine Learning Application - Genre Classification
# UDSL-SNU Big Data Academy   
# 20170725

# ##### Import libraries

# In[1]:

import h5py
import numpy as np
import matplotlib.pyplot as plt

from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn import svm
from sklearn.mixture import GaussianMixture


# ##### Load Data

# In[ ]:

with h5py.File('data/gtzan_mfcc.h', 'r') as f:
    X = np.asarray(f['X'])
    y = np.asarray(f['y'])
    genres = list(f['genres'])


# 1 audio clip has 120 dimensions   
# 60 features' mean and std   
# >60 features = 20(mfcc + delta_mfcc + double_deta_mfcc)

# In[ ]:

print('X.shape : {}'.format(X.shape))
print('y.shape : {}'.format(y.shape))


# In[ ]:

print('unique y : {}'.format(np.unique(y)))


# In[ ]:

plt.hist(y, range=[0,10])
plt.xlabel('y value')
plt.ylabel('count')
plt.title('Class distribution of GTZAN')
plt.show()


# ##### Train-test split

# In[ ]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y)


# ##### Linear SVM

# In[ ]:

linearSVM = svm.LinearSVC(C=.1, max_iter=100)
linearSVM.fit(X_train, y_train)


# Predicted value from SVM model

# In[ ]:

y_pred_SVM_train = linearSVM.predict(X_train)
y_pred_SVM = linearSVM.predict(X_test)


# In[ ]:

print ('(SVM)train acc : %f'% accuracy_score(y_train, y_pred_SVM_train))
print ('(SVM)test acc : %f'% accuracy_score(y_test, y_pred_SVM))
print ('(SVM)confusion matrix : ')
print (confusion_matrix(y_test, y_pred_SVM))


# ##### GMM model
# The GMM learns the statistical distribution of a particular instance,  
# in this case 10 GMMs are required (for 10 genres)

# In[ ]:

unique_class = range(10)
GMMs = dict() # array of GMM for each class

for c in unique_class : 
    GMMs[c] = GaussianMixture(n_components=32, covariance_type='full',
                                 tol = 0.05, reg_covar=3)
    index_gmm = np.where(y_train==c)[0]
    GMMs[c].fit(X_train[index_gmm])


# Scoring X using GMM

# In[ ]:

train_scores = list()
test_scores = list()
for c in unique_class : 
    train_scores.append(GMMs[c].score_samples(X_train))
    test_scores.append(GMMs[c].score_samples(X_test))

train_scores = np.asarray(train_scores).T
test_scores = np.asarray(test_scores).T


# In[ ]:

print ('train_scores.shape : {}'.format(train_scores.shape))


# find the index of the highest model

# In[ ]:

y_pred_GMM_train = np.argmax(train_scores, axis=1)
y_pred_GMM_test = np.argmax(test_scores, axis=1)


# In[ ]:

print ('(GMM)train acc : %f'% accuracy_score(y_train, y_pred_GMM_train))
print ('(GMM)test acc : %f'% accuracy_score(y_test, y_pred_GMM_test))
print ('(GMM)confusion matrix : ')
print (confusion_matrix(y_test, y_pred_GMM_test))


# ### Effect of Standardization

# In[ ]:

ss = StandardScaler()
ss.fit(X_train)
X_st_train = ss.transform(X_train)
X_st_test = ss.transform(X_test)


# ##### SVM

# In[ ]:

linearSVM = svm.LinearSVC(C=.1, max_iter=100)
linearSVM.fit(X_st_train, y_train)
y_pred_SVM_train = linearSVM.predict(X_st_train)
y_pred_SVM_st = linearSVM.predict(X_st_test)

print ('(SVM)train acc : %f'% accuracy_score(y_train, y_pred_SVM_train))
print ('(SVM)test acc : %f'% accuracy_score(y_test, y_pred_SVM_st))
print (confusion_matrix(y_test, y_pred_SVM_st))


# ##### GMM

# In[ ]:

unique_class = range(10) # array of GMM for each class

GMMs = dict()
for c in unique_class : 
    GMMs[c] = GaussianMixture(n_components=32, covariance_type='full',
                             tol = 0.05, reg_covar=3)
    index_gmm = np.where(y_train==c)[0]
    GMMs[c].fit(X_st_train[index_gmm])

# scoring X using GMM 
train_scores = list()
test_scores = list()
for c in unique_class : 
    train_scores.append(GMMs[c].score_samples(X_st_train))
    test_scores.append(GMMs[c].score_samples(X_st_test))

# find model shows best score
train_scores = np.asarray(train_scores).T
test_scores = np.asarray(test_scores).T

y_pred_GMM_train = np.argmax(train_scores, axis=1)
y_pred_GMM_test_st = np.argmax(test_scores, axis=1)

print ('(GMM)train acc : %f'% accuracy_score(y_train, y_pred_GMM_train))
print ('(GMM)test acc : %f'% accuracy_score(y_test, y_pred_GMM_test_st))
print ('(GMM)confusion matrix') 
print (confusion_matrix(y_test, y_pred_GMM_test))


# ### Results

# In[ ]:

print ('baseline')
print ('(SVM)test acc : %f'% accuracy_score(y_test, y_pred_SVM))
print ('(GMM)test acc : %f'% accuracy_score(y_test, y_pred_GMM_test))

print ('Standardization')
print ('(SVM)test acc : %f'% accuracy_score(y_test, y_pred_SVM_st))
print ('(GMM)test acc : %f'% accuracy_score(y_test, y_pred_GMM_test_st))

