
# coding: utf-8

# ### Machine Learning Application - Audio Feature Extraction
# UDSL-SNU Big Data Academy   
# 20170725

# ##### Import libraries

# In[1]:

import h5py
import numpy as np
import matplotlib.pyplot as plt


# 

# In[2]:




# ##### The data in the above code is as follows

# In[3]:

with h5py.File('data/audios.h', 'r') as f:
    x_blues = np.asarray(f['x_blues'])
    x_metal = np.asarray(f['x_metal'])
    x_rock = np.asarray(f['x_rock'])
    mfcc_blues = np.asarray(f['mfcc_blues'])
    mfcc_metal = np.asarray(f['mfcc_metal'])
    mfcc_rock = np.asarray(f['mfcc_rock'])
    sr = f['sr'].value


# ##### X-axis for plot

# In[4]:

ti_blues = [x / float(sr) for x in range(len(x_blues))]
ti_metal = [x / float(sr) for x in range(len(x_metal))]
ti_rock = [x / float(sr) for x in range(len(x_rock))]


# ##### Figure 1. Raw wave

# In[5]:

f, axarr = plt.subplots(3,1, sharex=True, sharey=True, figsize=(16,8))
f.suptitle('Waveform (raw data)')
axarr[0].plot(ti_blues, x_blues)
axarr[0].set_title('blues')
axarr[1].plot(ti_metal, x_metal)
axarr[1].set_title('metal')
axarr[2].plot(ti_rock, x_rock)
axarr[2].set_title('rock')
f.subplots_adjust(hspace=.5)
plt.xlim([0,30])
plt.xlabel('Time (s)')
plt.show()


# ##### Figure 2. Power Spectrum Density

# In[6]:

f, axarr = plt.subplots(3,1, sharex=True, sharey=True, figsize=(16,8))
f.suptitle('Power Spectrum Density')
axarr[0].psd(x_blues, NFFT=2048, Fs=sr)
axarr[0].set_title('blues')
axarr[1].psd(x_metal, NFFT=2048, Fs=sr)
axarr[1].set_title('metal')
axarr[2].psd(x_rock, NFFT=2048, Fs=sr)
axarr[2].set_title('rock')
f.subplots_adjust(hspace=.5)
plt.xlim([0, 11025])
plt.show()


# ##### Figure 3. Spectrogram

# In[7]:

f, axarr = plt.subplots(3,1, sharex=True, sharey=True, figsize=(16,8))
f.suptitle('Spectrogram')
axarr[0].specgram(x_blues, NFFT=2048, Fs=sr, noverlap=0)
axarr[0].set_title('blues')
axarr[1].specgram(x_metal, NFFT=2048, Fs=sr, noverlap=0)
axarr[1].set_title('metal')
axarr[2].specgram(x_rock, NFFT=2048, Fs=sr, noverlap=0)
axarr[2].set_title('rock')
f.subplots_adjust(hspace=.5)
plt.xlim([0,30])
plt.xlabel('Time (s)')
plt.ylim([0, 11025])
plt.show()


# ##### Figure 4. MFCCs

# In[8]:

f, axarr = plt.subplots(3,1, sharex=True, sharey=True, figsize=(16,8))
f.suptitle('Mel-Frequency Cepstrum Coefficients (MFCCs)')
im1 = axarr[0].imshow(mfcc_blues, aspect='auto', interpolation='nearest')
axarr[0].set_title('blues')
plt.colorbar(im1, ax=axarr[0])
im2 = axarr[1].imshow(mfcc_metal, aspect='auto', interpolation='nearest')
axarr[1].set_title('metal')
plt.colorbar(im2, ax=axarr[1])
im3 = axarr[2].imshow(mfcc_rock, aspect='auto', interpolation='nearest')
axarr[2].set_title('rock')
plt.colorbar(im3, ax=axarr[2])
f.subplots_adjust(hspace=.5)

plt.xlabel('Frame')
plt.show()


# In[ ]:



