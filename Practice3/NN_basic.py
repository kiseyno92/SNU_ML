
# coding: utf-8

# In[9]:

import tensorflow as tf
import numpy as np
x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype = np.float32)


# In[10]:

y_data = np.array([[0],[1],[1],[0]], dtype = np.float32)


X = tf.placeholder(tf.float32, [None,2])
Y = tf.placeholder(tf.float32, [None,1])





# In[11]:

W1 = tf.Variable(tf.random_normal([2,2]), name = 'weight1')
b1 = tf.Variable(tf.random_normal([2]), name = 'bias1')
layer1 = tf.sigmoid(tf.matmul(X,W1)+b1)


# In[12]:

W2 = tf.Variable(tf.random_normal([2,1]), name = 'weight2')
b2 = tf.Variable(tf.random_normal([1]), name = 'bias2')


# In[13]:

hypothesis = tf.sigmoid(tf.matmul(layer1,W2)+b2)


# In[14]:

cost = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))


# In[18]:

train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)


# In[19]:

predicted = tf.cast(hypothesis>0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y),dtype = tf.float32))


# In[20]:

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(2000):
        sess.run(train, feed_dict = {X:x_data, Y: y_data})
        if step %100 ==0:
            print(step, sess.run(cost,feed_dict = {X:x_data, Y:y_data}), sess.run([W1,W1]))
    h,p,a = sess.run([hypothesis, predicted, accuracy], feed_dict ={X:x_data, Y: y_data})
    print ('\nHypothsis:',h,'\npredicted:',p,'\naccuracy:',a)


# In[ ]:



