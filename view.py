import numpy as np # linear algebra
np.random.seed(666)
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import cv2

from pyradar.filters.frost import frost_filter
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance

from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
from sklearn.model_selection import train_test_split
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)

from keras.preprocessing.image import ImageDataGenerator
#for rotating pictures
datagen = ImageDataGenerator(rotation_range=90)

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


#adjust for the incident angle and use the magnitude
# for idx, band in enumerate(x_band1):
# 	x_band1[idx] = (np.multiply(10**(band/20), np.sin(X_angle_train[idx]*3.14159/180)/np.cos(X_angle_train[idx]*3.14159/180)))
# for idx, band in enumerate(x_band2):
# 	x_band2[idx] = (np.multiply(10**(band/20), np.sin(X_angle_train[idx]*3.14159/180)/np.cos(X_angle_train[idx]*3.14159/180)))

#calculate gradient band
grad_band1 = x_band1
grad_band2 = x_band2
for idx, band in enumerate(grad_band1):
	grad = np.gradient(band)
	m_grad = np.sqrt(grad[0]**2 + grad[1]**2)
	grad_band1[idx] = m_grad
for idx, band in enumerate(grad_band2):
	grad = np.gradient(band)
	m_grad = np.sqrt(grad[0]**2 + grad[1]**2)
	grad_band2[idx] = m_grad

print("done2")
y_train = np.array(train["is_iceberg"])
y_train = y_train.reshape(y_train.size,1)

print x_band1[:, :, :, np.newaxis].shape
print x_band2[:, :, :, np.newaxis].shape
print grad_band1[:, :, :, np.newaxis].shape
print grad_band2[:, :, :, np.newaxis].shape

X_train = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis],grad_band1[:, :, :, np.newaxis],grad_band2[:, :, :, np.newaxis]], axis = -1)

print np.shape(x_band1)
print np.shape(x_band2)
print np.shape(X_train)
print np.shape(X_angle_train)
print np.shape(y_train)

X_train, X_valid, X_angle_train, X_angle_valid, y_train, y_valid = train_test_split(X_train,X_angle_train,y_train, train_size=0.75)


from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.constraints import maxnorm
from keras.layers.merge import Concatenate
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

def get_callbacks(filepath, patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]
    
def get_model():
    bn_model = 0
    p_activation = "elu"
    input_1 = Input(shape=(75, 75, 2), name="X_1_lee")
    input_2 = Input(shape=[75, 75, 2], name="X_1_grad")
    
    #did pretty well ~0.17 log loss
    # img_1 = Conv2D(16, kernel_size = (3,3), activation=p_activation) ((BatchNormalization(momentum=bn_model))(input_1))
    # img_1 = MaxPooling2D((2,2)) (img_1)
    # img_1 = Dense(64, activation = 'relu') (img_1)
    # img_1 = Dropout(0.3)(img_1)
    # img_1 = Conv2D(32, kernel_size = (3,3), activation=p_activation) (img_1)
    # img_1 = MaxPooling2D((2,2)) (img_1)
    # img_1 = Dense(128, activation = 'relu') (img_1)
    # img_1 = Dropout(0.3)(img_1)
    # img_1 = Conv2D(64, kernel_size = (3,3), activation=p_activation) (img_1)
    # img_1 = MaxPooling2D((2,2)) (img_1)
    # img_1 = Dense(256, activation = 'relu') (img_1)
    # img_1 = Dropout(0.3)(img_1)
    # img_1 = GlobalMaxPooling2D() (img_1)

    #first image
    img_1 = Conv2D(16, kernel_size = (3,3), activation=p_activation) ((BatchNormalization(momentum=bn_model))(input_1))
    img_1 = MaxPooling2D((2,2)) (img_1)
    img_1 = Dense(64, activation = 'relu') (img_1)
    img_1 = Dropout(0.3)(img_1)
    img_1 = Conv2D(32, kernel_size = (3,3), activation=p_activation) (img_1)
    img_1 = MaxPooling2D((2,2)) (img_1)
    img_1 = Dense(128, activation = 'relu') (img_1)
    img_1 = Dropout(0.3)(img_1)
    img_1 = Conv2D(64, kernel_size = (3,3), activation=p_activation) (img_1)
    img_1 = MaxPooling2D((2,2)) (img_1)
    img_1 = Dense(256, activation = 'relu') (img_1)
    img_1 = Dropout(0.3)(img_1)
    img_1 = MaxPooling2D((2,2)) (img_1)

    img_2 = Conv2D(16, kernel_size = (3,3), activation=p_activation) ((BatchNormalization(momentum=bn_model))(input_2))
    img_2 = MaxPooling2D((2,2)) (img_2)
    img_2 = Dense(64, activation = 'relu') (img_2)
    img_2 = Dropout(0.3)(img_2)
    img_2 = Conv2D(32, kernel_size = (3,3), activation=p_activation) (img_2)
    img_2 = MaxPooling2D((2,2)) (img_2)
    img_2 = Dense(128, activation = 'relu') (img_2)
    img_2 = Dropout(0.3)(img_2)
    img_2 = Conv2D(64, kernel_size = (3,3), activation=p_activation) (img_2)
    img_2 = MaxPooling2D((2,2)) (img_2)
    img_2 = Dense(256, activation = 'relu') (img_2)
    img_2 = Dropout(0.3)(img_2)
    img_2 = MaxPooling2D((2,2)) (img_2)

    print img_1.shape
    print img_2.shape
    combined_imgs = Concatenate()([img_1,img_2])

    combined = Dense(16, activation = 'relu') (combined_imgs)
    combined = Dropout(0.3)(combined)
    combined = Flatten() (combined)

    # img_1 = Conv2D(16, kernel_size = (3,3), activation=p_activation) ((BatchNormalization(momentum=bn_model))(input_1))
    # img_1 = Conv2D(24, kernel_size = (3,3), activation=p_activation) (img_1)
    # img_1 = Conv2D(32, kernel_size = (3,3), activation=p_activation) (img_1)
    # img_1 = Conv2D(24, kernel_size = (3,3), activation=p_activation) (img_1)
    # img_1 = Conv2D(24, kernel_size = (3,3), activation=p_activation) (img_1)
    # img_1 = MaxPooling2D((2,2)) (img_1)
    # img_1 = Dense(256, activation = 'relu') (img_1)
    # img_1 = Dropout(0.3)(img_1)
    # img_1 = Dense(256, activation = 'relu') (img_1)
    # img_1 = Dropout(0.3)(img_1)
    # img_1 = GlobalMaxPooling2D() (img_1)
    output = Dense(1, activation="sigmoid")(combined)
    
    model = Model([input_1,input_2],  output)
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

model = get_model()
print(model.summary())

file_path = ".model_weights.hdf5"
callbacks = get_callbacks(filepath=file_path, patience=5)

print X_train[:,:,:,:2].shape
print X_train[:,:,:,2:4].shape
print y_train.shape
print y_valid.shape

model.fit([X_train[:,:,:,:2],X_train[:,:,:,2:4]], y_train, epochs=25
          , validation_data=([X_valid[:,:,:,:2], X_valid[:,:,:,2:4]], y_valid)
         , batch_size=32
         , callbacks=callbacks)

# print("IS BERG?")
# w = 6

# print(y_train[w])
# band1 = x_band1[w]
# band2 = x_band2[w]
# angle = X_angle_train[w]

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