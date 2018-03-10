import numpy as np # linear algebra
np.random.seed(666)
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
# import cv2

from pyradar.filters.frost import frost_filter
from pyradar.filters.lee_enhanced import lee_enhanced_filter
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance

from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral, denoise_nl_means,
                                 denoise_wavelet, estimate_sigma)
from keras.models import load_model

from keras.preprocessing.image import ImageDataGenerator
import cv2

def lee_filter(img, size):
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance**2 / (img_variance**2 + overall_variance**2)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output

train = pd.read_json("./data/processed/train.json")
train.inc_angle = train.inc_angle.replace('na', 23)

print("done")

X_angle_train = np.array(train.inc_angle)
X_angle_train = X_angle_train.reshape(X_angle_train.size, 1)
x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])


lee_filtered1 = np.load("lee_filtered1.npy")
lee_filtered2 = np.load("lee_filtered2.npy")

# #calculate gradient band
grad_band1 = np.array(x_band1)
grad_band2 = np.array(x_band2)
for idx, band in enumerate(grad_band1):
	grad = np.gradient(band)
	m_grad = np.sqrt(grad[0]**2 + grad[1]**2)
	grad_band1[idx] = m_grad
for idx, band in enumerate(grad_band2):
	grad = np.gradient(band)
	m_grad = np.sqrt(grad[0]**2 + grad[1]**2)
	grad_band2[idx] = m_grad

# print("done2")
y_train = np.array(train["is_iceberg"])
y_train = y_train.reshape(y_train.size,1)

# print x_band1[:, :, :, np.newaxis].shape
# print x_band2[:, :, :, np.newaxis].shape
# print grad_band1[:, :, :, np.newaxis].shape
# print grad_band2[:, :, :, np.newaxis].shape

avg_band = (x_band1+x_band2)/2
avg_band = (np.sqrt(x_band1**2 + x_band2**2))

X_train = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis],grad_band1[:,:,:,np.newaxis], grad_band2[:,:,:,np.newaxis]], axis = -1)

# print np.shape(x_band1)
# print np.shape(x_band2)
# print np.shape(X_train)
# print np.shape(X_angle_train)
# print np.shape(y_train)

# X_train, X_valid, X_angle_train, X_angle_valid, y_train, y_valid = train_test_split(X_train,X_angle_train,y_train, train_size=0.95)

print("IS BERG?")
w = 1224

# # print(y_train[w])
print x_band1[w]
print x_band1[w].max()

band1 = np.abs(x_band1[w])/max(x_band1[w])
band2 = np.abs(x_band2[w])/max(x_band2[w])
# band1 = band1.reshape(band1.size,1)
# band2 = band2.reshape(band2.size,1)
# noise1 = denoise_tv_chambolle(x_band1[w], weight=0.25, multichannel=True)
# noise2 = denoise_tv_chambolle(x_band2[w], weight=0.25, multichannel=True)

# band1 = (band1-band1.min())/(band1.max()-band1.min())
# band2 = (band2-band2.min())/(band2.max()-band2.min())

print band1
print band2 

noise_nl_1 = denoise_nl_means(band1)
noise_nl_2 = denoise_nl_means(band2)
noise1 = denoise_bilateral(band1,multichannel=False)
noise2 = denoise_bilateral(band2,multichannel=False)

plt.subplot(3,3,1)
plt.imshow(band1 , cmap='gray')
plt.subplot(3,3,2)
plt.imshow(band2, cmap='gray')
plt.subplot(3,3,3)
plt.imshow(noise1 , cmap='gray')
plt.subplot(3,3,4)
plt.imshow(noise2 , cmap='gray')
plt.subplot(3,3,5)
plt.imshow(noise_nl_1 , cmap='gray')
plt.subplot(3,3,6)
plt.imshow(noise_nl_2 , cmap='gray')
plt.show()
