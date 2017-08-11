
# coding: utf-8

# ### Handling the realworld data - Linear Regression
# UDSL-SNU Big Data Academy   
# 20170725

# ##### Import libraries

# In[ ]:

import matplotlib.pyplot as plt
import numpy as np
import h5py
from sklearn import linear_model


# ##### Load data

# In[ ]:

with h5py.File('data/weight_cost.h', 'r') as f: 
    X = np.asarray(f['X'])
    y = np.asarray(f['y'])


# ##### Linear regression using Sklearn implementation

# library need shape (n,1) not (n,)

# In[ ]:

X = X.reshape((X.shape[0], 1))
y = y.reshape((y.shape[0], 1))


# Model fitting

# In[ ]:

regModel = linear_model.LinearRegression()
regModel.fit(X, y)


# Model without outliers

# In[ ]:

index_normal = np.where(X > 200)[0]
regModel_norm = linear_model.LinearRegression()
regModel_norm.fit(X[index_normal], y[index_normal])


# ##### plot data

# x axis linspace for plot

# In[ ]:

x_axis = range(700)
x_axis = np.asarray(x_axis).reshape((len(x_axis),1))


# In[ ]:

plt.figure()
plt.scatter(X, y)
plt.plot(x_axis, regModel.predict(x_axis), color='red')
plt.plot(x_axis, regModel_norm.predict(x_axis), color='orange')

plt.title('Linear Regression (slope = %0.2f, %0.2f)'%(regModel.coef_, regModel_norm.coef_) )
plt.xlabel('Weight (gram)', fontsize = 15)
plt.ylabel('Cost (Won)', fontsize = 15)
plt.show()

