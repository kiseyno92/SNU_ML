
# coding: utf-8

# In[1]:

import keras 
import numpy as np


# In[2]:

from keras.datasets import boston_housing, mnist , cifar10, imdb


# In[4]:

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()


# In[5]:

print(x_train.shape)
print(x_test.shape)


# In[7]:

print(y_train.shape)
print(y_test.shape)


# In[8]:

from keras.models import Sequential # model 이름 t순차적으로 들어 간다는 뜻
from keras.layers import Dense  # fully connected layer
from keras import losses
from sklearn.metrics import mean_squared_error


# In[26]:

model = Sequential()
model.add(Dense(13, input_dim =x_train.shape[1], kernel_initializer = 'normal', activation ='relu'))
model.add(Dense(9,kernel_initializer = 'normal', activation ='relu'))
model.add(Dense(7,kernel_initializer = 'normal', activation ='relu'))
model.add(Dense(3,kernel_initializer = 'normal', activation ='relu'))
model.add(Dense(1,kernel_initializer = 'normal'))
#텐서플로는 bias등 여러가지 설정을 해주어야 하지만 keras는 그냥 모델만 설정해주면 간편히 만든다.


# In[27]:

model.compile(loss = 'mean_squared_error', optimizer = 'adam')


# In[28]:

model.fit(x_train, y_train, batch_size =30, epochs = 15, verbose =1)


# In[18]:

y_pred = model.predict(x_test)


# In[19]:

mean_squared_error(y_pred, y_test)


# In[ ]:



