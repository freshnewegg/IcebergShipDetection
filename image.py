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
#for rotating pictures

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


# normalize the magnitude
# for idx,row in enumerate(val):
#     for idy,col in enumerate(row):
#         if col > 1:
#             x_band1[idx][idy] = (1 + np.log(col))/max(col, 1+np.log(col))
#         elif col > 0 and col <=1:
#             x_band1[idx][idy] = (col/(max(col,1+np.log(col))))
#         else:
#             x_band1[idx][idy] = col

# for idx,row in enumerate(val):
#     for idy,col in enumerate(row):
#         if col > 1:
#             x_band2[idx][idy] = (1 + np.log(col))/max(col, 1+np.log(col))
#         elif col > 0 and col <=1:
#             x_band2[idx][idy] = (col/(max(col,1+np.log(col))))
#         else:
#             x_band1[idx][idy] = col

# adjust for the incident angle and use the magnitude
# for idx, band in enumerate(x_band1):
# 	# x_band1[idx] = (np.multiply(10**(band/20), np.sin(X_angle_train[idx]*3.14159/180)/np.cos(X_angle_train[idx]*3.14159/180)))
#     # x_band1[idx]/=np.cos(X_angle_train[idx]*3.14159/180)
#     x_band1[idx] = lee_enhanced_filter(np.abs(band))
#     if idx%10:
#         print idx
# for idx, band in enumerate(x_band2):
#     x_band2[idx] = lee_enhanced_filter(np.abs(band))
#     if idx%10:
#         print idx
#     # x_band2[idx]/=np.cos(X_angle_train[idx]*3.14159/180)
# 	# x_band2[idx] = (np.multiply(10**(band/20), np.sin(X_angle_train[idx]*3.14159/180)/np.cos(X_angle_train[idx]*3.14159/180)))

# np.save("lee_filtered1", x_band1)
# np.save("lee_filtered2",x_band2)
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
w = 1473

# # print(y_train[w])
print x_band1[w]
print x_band1[w].max()

band1 = x_band1[w]
band2 = x_band2[w]
# band1 = band1.reshape(band1.size,1)
# band2 = band2.reshape(band2.size,1)
# noise1 = denoise_tv_chambolle(x_band1[w], weight=0.25, multichannel=True)
# noise2 = denoise_tv_chambolle(x_band2[w], weight=0.25, multichannel=True)

band1 = (band1-band1.min())/(band1.max()-band1.min())
band2 = (band2-band2.min())/(band2.max()-band2.min())

print band1
print band2 

noise_nl_1 = denoise_nl_means(band1)
noise_nl_2 = denoise_nl_means(band2)
noise1 = denoise_bilateral(band1,multichannel=False)
noise2 = denoise_bilateral(band2,multichannel=False)
# noise1 = cv2.fastNlMeansDenoising(band1, None,10)
# noise2 = cv2.fastNlMeansDenoising(band2, None,10)
# angle = X_angle_train[w]

# band1_min_val = np.amin(band1)
# band1_max_val = np.amax(band1)9 

# band2_min_val = np.amin(band2)
# band2_max_val = np.amax(band2)


# normal1 = np.array(band1)
# normal2 = np.array(band2)

# # normalize the magnitude
# for idx,row in enumerate(normal1):
#     for idy,col in enumerate(row):
#         if col > 1:
#             normal1[idx][idy] *= (1 + np.log(col))/max(col, 1+np.log(col))
#         else:
#             normal1[idx][idy] *= (col/(max(col,1+np.log(col))))

# for idx,row in enumerate(normal2):
#     for idy,col in enumerate(row):
#         if col > 1:
#             normal2[idx][idy] *= (1 + np.log(col))/max(col, 1+np.log(col))
#         else:
#             normal2[idx][idy] *= (col/(max(col,1+np.log(col))))


# use some factor from the papaer
# magnitude[magnitude>1]*=(1+np.log(magnitude[magnitude>1]))/(max(1+np.log(magnitude[magnitude>1]),magnitude[magnitude>1]))
# magnitude[magnitude<=1]*=magnitude[magnitude>1]/(max(1+np.log(magnitude[magnitude>1]),magnitude[magnitude>1]))

# magnitude2[magnitude2>1]*=(1+np.log(magnitude2[magnitude2>1]))/(max(1+np.log(magnitude2[magnitude2>1]),magnitude2[magnitude2>1]))
# magnitude2[magnitude2<=1]*=magnitude2[magnitude2>1]/(max(1+np.log(magnitude2[magnitude2>1]),magnitude2[magnitude2>1]))

# band1 = 255*(band1 - np.min(band1))/(np.max(band1)-np.min(band1))
# band2 = 255*(band2 - np.min(band2))/(np.max(band2)-np.min(band2))

# model = load_model('.model_weights.hdf5')

# prediction = model.predict([X_train[:,:,:,:2],X_train[:,:,:,2:4]], verbose=1, batch_size=200)
# y_classes = prediction.argmax(axis=-1)
# prediction = np.squeeze(prediction)
# prediction[prediction==0] = 1e-10
# print prediction
# print np.squeeze(y_train)
# print log_loss(y_train, prediction)

# incorrects = np.nonzero(prediction != y_train)
# bad_indexes = np.load("difficult_indexes.npy")

# print prediction[1508]
# print y_train[1508]
# print bad_indexes
# for bad in bad_indexes:
#     print("idx: "+ str(bad) + " predicted: " + str(prediction[bad]) + ', was: ' + str(y_train[bad]))
# for idx,x in enumerate(bad_indexes[10:15]):
#     w = x
#     idx+=1
#     # print(y_train[w])
#     # print(x)
#     band1 = 10**(x_band1[w]/20)
#     band2 = 10**(x_band2[w]/20)
#     angle = X_angle_train[w]
#     print band1.shape
#     plt.subplot(5,2,idx)
#     plt.imshow(band1 , cmap='gray')
#     plt.subplot(5,2,idx+5)
#     plt.imshow(band2 , cmap='gray')

# plt.show()


# print len(incorrects[1])

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
# print(band1)
# band1*=163000*np.sqrt(1/np.cos(angle)**2 - 1)
# print(band1)
plt.subplot(3,3,6)
plt.imshow(noise_nl_2 , cmap='gray')
# band2*=163000*np.sqrt(1/np.cos(angle)**2 - 1)
# plt.subplot(3,3,6)
# plt.imshow(band2 , cmap='gray')
# fband2 = lee_enhanced_filter(np.abs(band2))
# plt.imshow(fband2, cmap='gray')
# plt.subplot(3,3,5)
# plt.imshow(np.sqrt(fband1**2 + fband2**2), cmap='gray')
# # band1 = band1[:,:,np.newaxis]
# # band2 = band2[:,:,np.newaxis]
# # print band1
plt.show()