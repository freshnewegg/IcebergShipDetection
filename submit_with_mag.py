import pandas as pd
from pandas import read_csv
import numpy as np
from pyradar.filters.frost import frost_filter
from pyradar.filters.lee_enhanced import lee_enhanced_filter
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
from sklearn.model_selection import train_test_split
from keras.models import load_model

def lee_filter(img, size):
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance**2 / (img_variance**2 + overall_variance**2)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output

test = pd.read_json("./data/processed/test.json")
x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])

# for idx, band in enumerate(x_band1):
# 	# x_band1[idx] = (np.multiply(10**(band/20), np.sin(X_angle_train[idx]*3.14159/180)/np.cos(X_angle_train[idx]*3.14159/180)))
#     # x_band1[idx]/=np.cos(X_angle_train[idx]*3.14159/180)
#     x_band1[idx] = lee_enhanced_filter(band + 83)
#     if not idx%10:
#         print idx
# for idx, band in enumerate(x_band2):
#     x_band2[idx] = lee_enhanced_filter(band + 83)
#     if not idx%10:
#         print idx

# np.save("test_lee_filtered1", x_band1)
# np.save("test_lee_filtered2",x_band2)

lee_filtered1 = np.load("test_lee_filtered1.npy")
lee_filtered2 = np.load("test_lee_filtered2.npy")

X_angle_test = np.array(test.inc_angle)
avg_band = (x_band1+x_band2)/2

#calculate gradient band
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

x_band1 = np.load("test_normalized_band1.npy")
x_band2 = np.load("test_normalized_band2.npy")

X_test = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis], grad_band1[:,:,:,np.newaxis], grad_band2[:,:,:,np.newaxis]], axis = -1)

model = load_model('.model_weights.hdf5')

prediction = model.predict([X_test[:,:,:,:2],X_test[:,:,:,2:4]], verbose=1, batch_size=200)

submission = pd.DataFrame({'id': test["id"], 'is_iceberg': prediction.reshape((prediction.shape[0]))})

submission.to_csv("./submission.csv", index=False)

# print("IS BERG?")
# w = 142

# # print(y_test[w])
# band1 = x_band1[w]
# band2 = x_band2[w]
# angle = X_angle_test[w]

# magnitude = band1#10**(band1/20) * math.sin(angle*3.14159/180)/math.cos(angle*3.14159/180)
# magnitude2 = band2#10**(band2/20) * math.sin(angle*3.14159/180)/math.cos(angle*3.14159/180)


# #use some factor from the papaer
# # magnitude[magnitude>1]*=(1+np.log(magnitude[magnitude>1]))/(max(1+np.log(magnitude[magnitude>1]),magnitude[magnitude>1]))
# # magnitude[magnitude<=1]*=magnitude[magnitude>1]/(max(1+np.log(magnitude[magnitude>1]),magnitude[magnitude>1]))

# # magnitude2[magnitude2>1]*=(1+np.log(magnitude2[magnitude2>1]))/(max(1+np.log(magnitude2[magnitude2>1]),magnitude2[magnitude2>1]))
# # magnitude2[magnitude2<=1]*=magnitude2[magnitude2>1]/(max(1+np.log(magnitude2[magnitude2>1]),magnitude2[magnitude2>1]))

# plt.subplot(2,3,1)
# grad = np.gradient(magnitude)
# grad = np.sqrt(grad[0]**2 + grad[1]**2)
# print grad.shape
# plt.imshow(grad , cmap='gray')
# plt.subplot(2,3,2)
# plt.imshow(magnitude, cmap='gray')
# # plt.subplot(2,3,3)
# # plt.imshow(cv2.bilateralFilter(magnitude,9,75,75), cmap='gray')
# plt.subplot(2,3,4)
# grad = np.gradient(magnitude2)
# grad = np.sqrt(grad[0]**2 + grad[1]**2)
# plt.imshow(grad, cmap='gray')
# plt.subplot(2,3,5)
# plt.imshow(magnitude2, cmap='gray')
# # plt.subplot(2,3,6)
# # plt.imshow(cv2.bilateralFilter(magnitude2,9,75,75), cmap='gray')
# plt.show()