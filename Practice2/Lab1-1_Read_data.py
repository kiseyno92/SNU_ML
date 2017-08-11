
# coding: utf-8

# ### Handling the realworld data - read data
# UDSL-SNU Big Data Academy  
# 2017.07.25  

# ##### Import libraries

# In[1]:

import csv
import matplotlib.pyplot as plt
import numpy as np


# ##### Open & Read csv file
# with ***CLEAN*** data

# In[2]:

data_clean = np.loadtxt('data/raw_data_clean.csv', dtype='string')
for row in data_clean : 
    print row


# with ***REALWORLD*** data

# In[3]:

data_real = np.loadtxt('data/raw_data.csv', dtype='string')


# ##### We need to check the raw csv file!!

# In[ ]:

cntRow = 0 # for checking the number of rows
csvLength = list() # for checking the column of each row
data_real = list() # for saving raw data

with open('data/raw_data.csv', 'r') as csvFile : 
    csvReader = csv.reader(csvFile, delimiter=',')
    for row in csvReader : 
        print(row)
        data_real.append(row)
        csvLength.append(len(row))
        cntRow = cntRow + 1


# ##### check the data details

# In[ ]:

print('There are {} rows in csv file'.format(cntRow))


# In[ ]:

print('The rows in the column are :')
print(csvLength)


# In[ ]:

print("The list's unique numbers are : ")
print(np.unique(csvLength))


# In[ ]:

plt.figure()
plt.plot(csvLength)
plt.title('len(csvLength)')
plt.ylim([3,7])
plt.xlabel('row')
plt.ylabel('count')
plt.show()


# ##### find abnormal data points

# In[ ]:

index_abnormal = np.where(np.asarray(csvLength)==6)[0]
print ('The index of an abnormal data point is : ')
print (index_abnormal)


# In[ ]:

print ('Header : {}'.format(data_real[0])) # header of csv
for idx in index_abnormal : 
    print (data_real[idx])

