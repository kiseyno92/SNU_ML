
# coding: utf-8

# In[1]:

import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(777)


# In[2]:

x_train = [1,2,3]
y_train = [1,2,3]


# In[3]:

plt.plot(x_train, y_train)
plt.grid()
plt.show()


# In[4]:

W = tf.Variable(tf.random_normal([1]), name = 'Weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')


# In[5]:

hypothesis = x_train *W +b


# In[6]:

cost = tf.reduce_mean(tf.square(hypothesis - y_train))


# In[8]:

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)


# In[9]:

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        sess.run(train)
        if step%100 == 0:
            print(step, sess.run(cost),sess.run(W),sess.run(b))


# In[ ]:



