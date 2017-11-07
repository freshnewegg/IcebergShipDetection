import numpy as np # linear algebra
np.random.seed(666)
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math

from pyradar.filters.frost import frost_filter
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance

from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
from sklearn.model_selection import train_test_split
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)

from keras.preprocessing.image import ImageDataGenerator


from matplotlib import pyplot
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras import backend as K
from keras.constraints import maxnorm
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.merge import Concatenate
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

from sklearn.model_selection import train_test_split
from keras.models import load_model


model = load_model('.model_weights.hdf5')
train = pd.read_json("./data/processed/train.json")
X_angle_train = np.array(train.inc_angle)
X_angle_train = X_angle_train.reshape(X_angle_train.size, 1)

x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])



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

x_band1 = np.load("normalized_band1.npy")
x_band2 = np.load("normalized_band2.npy")

y_train = np.array(train["is_iceberg"])
y_train = y_train.reshape(y_train.size,1)
X_train = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis],grad_band1[:,:,:,np.newaxis], grad_band2[:,:,:,np.newaxis]], axis = -1)

prediction = model.predict([X_train[:,:,:,:2],X_train[:,:,:,2:4]], verbose=1, batch_size=200)

print "PRESQUEEZE"
print prediction[1508]

prediction[prediction==1] = 1 - 1e-10
prediction = np.squeeze(prediction)

y_train_f = np.squeeze(y_train)

wrong = []

print y_train_f[1508]
print prediction[1508]

for idx, x in enumerate(prediction):
	if np.abs(y_train_f[idx]-x) > 0.5:
		# print y_train_f[idx]
		# print x
		# print np.abs(y_train_f[idx]-x)
		# print idx
		wrong.append(idx)

print len(wrong)
for bad in wrong:
    print("idx: "+ str(bad) + " predicted: " + str(prediction[bad]) + ', was: ' + str(y_train[bad]))

print K.eval((binary_crossentropy(K.variable(y_train_f), K.variable(prediction))))

