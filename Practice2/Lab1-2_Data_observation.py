
# coding: utf-8

# ### Handling the realworld data - Data observation
# UDSL-SNU Big Data Academy  
# 20170725  

# ##### Import libraries

# In[ ]:

import csv
import matplotlib.pyplot as plt
import numpy as np
import h5py


# ##### List variables for saving columns

# In[ ]:

weights = list()
places = list()
costs = list()
customers = list()
scales = list()


# ##### Open csv file and save column
# when the number of column is 6, the second and third column mean ***place***

# In[ ]:

with open('data/raw_data.csv', 'r') as csvFile : 
    csvReader = csv.reader(csvFile, delimiter=',')
    for row in csvReader : 
        if len(row) == 5: 
            weights.append(row[0])
            places.append(row[1])
            costs.append(row[2])
            customers.append(row[3])
            scales.append(row[4])
        else :
            weights.append(row[0])
            places.append(row[1]+','+row[2])
            costs.append(row[3])
            customers.append(row[4])
            scales.append(row[5])


# In[ ]:

print ('Weights : ')
print (weights)


# In[ ]:

print ('Places :')
print (places)


# In[ ]:

print ('Costs :')
print (costs)


# In[ ]:

print ('Customers :')
print (customers)


# In[ ]:

print ('Scales :')
print (scales)


# ##### Remove headers

# In[ ]:

weights = weights[1:]
places = places[1:]
costs = costs[1:]
customers = customers[1:]
scales = scales[1:]


# ##### check the string variables

# In[ ]:

print('Unique places :')
print(np.unique(places))

print('Unique customers :')
print(np.unique(customers))

print('Unique scales :')
print(np.unique(scales))


# ##### check the numeric variables

# In[ ]:

np.mean(costs)


# ##### Convert string to numeric

# In[ ]:

weights = [float(x) for x in weights]
costs = [float(x) for x in costs]


# In[ ]:

print ('Weights :')
print ('mean({}), std({}), max({}), min({})'.format(np.mean(weights),np.std(weights), 
                                                    np.max(weights), np.min(weights)))


# In[ ]:

print ('Costs :')
print ('mean({}), std({}), max({}), min({})'.format(np.mean(costs),np.std(costs), 
                                                    np.max(costs), np.min(costs)))


# ##### Plot numeric variables

# In[ ]:

plt.figure()
plt.scatter(weights, costs)
plt.title('weight-cost scatter plot')
plt.xlabel('weight')
plt.ylabel('cost')
plt.show()


# ##### Check outlier points
# need numpy-array format to find element index

# In[ ]:

weights = np.asarray(weights)
costs = np.asarray(costs)

index_normal = np.where(weights>200)[0]
index_outlier = np.where(weights<200)[0]

print('There are %d outliers in data'%(len(index_outlier)))


# In[ ]:

plt.figure()
plt.scatter(weights[index_normal], costs[index_normal])
plt.title('weight-cost w/o outliers')
plt.xlabel('weight')
plt.ylabel('cost')
plt.show()


# ##### Plot data with constraint '*SCALE*'

# In[ ]:

scales = np.asarray(scales)
index_a_scale = np.where(scales=='a')[0]
index_d_scale = np.where(scales=='d')[0]

f, axarr = plt.subplots(2,1, sharex=True)
axarr[0].scatter(weights[index_a_scale], costs[index_a_scale])
axarr[1].scatter(weights[index_d_scale], costs[index_d_scale])
f.suptitle('scale-wise plot')
axarr[0].set_title('a scale')
axarr[1].set_title('d scale')
plt.show()

