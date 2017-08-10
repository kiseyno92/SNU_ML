
# coding: utf-8

# Linear Regression Example

# In[1]:

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model


# In[4]:

#Load the diabetes dataset
diabetes = datasets.load_diabetes()
#diabetes


# In[6]:

diabetes_X = diabetes.data[:,np.newaxis, 2]#모든 행에 대해서 2 index 라인을 사용하겠다


# In[10]:

# Split the data into training/testing sets 처음부터 뒤에서 20명까지
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

#Split the targets into sets
diabetes_Y_train = diabetes.target[:-20]
diabetes_Y_test = diabetes.target[-20:]


# In[11]:

#Plot train data
plt.scatter(diabetes_X_train, diabetes_Y_train, color = 'black')
plt.title('diabetes_train')
plt.xlabel('BMI')
plt.ylabel('diabetes')
plt.show()


# In[13]:

#Create linear regression object
regr = linear_model.LinearRegression()

#Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_Y_train)


# In[14]:

print('Coefficient: \n', regr.coef_)
print('Intercept: \n', regr.intercept_)


# In[15]:

def linearFunction(x,a,b):
    y = (a*x) + b
    return y


# In[19]:

plt.scatter(diabetes_X_train,diabetes_Y_train, color = 'black')

x = np.arange(-0.1,0.2,0.1) #x값 넣어주기
y = linearFunction(x,regr.coef_,regr.intercept_)
plt.plot(x,y,color = 'blue',linewidth =3)

plt.title('diabetes_train')
plt.xlabel('BMI')
plt.ylabel('diabetes')
plt.show()


# Test Model

# In[20]:

plt.scatter(diabetes_X_test,diabetes_Y_test, color = 'black')
plt.plot(diabetes_X_test,regr.predict(diabetes_X_test),color = 'blue',linewidth =3)

plt.title('diabetes_test')
plt.xlabel('BMI')
plt.ylabel('diabetes')
plt.show()


# In[24]:

#The mean squared error
print ("Mean squared error : %.2f" %np.mean((regr.predict(diabetes_X_test) - diabetes_Y_test)**2))

print("Variance score : %.2f" % regr.score(diabetes_X_test, diabetes_Y_test))
      


# In[ ]:




# In[ ]:



