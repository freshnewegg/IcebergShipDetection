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
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,denoise_nl_means,
                                 denoise_wavelet, estimate_sigma)

from keras.preprocessing.image import ImageDataGenerator
#for rotating pictures
datagen = ImageDataGenerator(
    horizontal_flip = True,
    vertical_flip=True
    )

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
# grad_band1 = np.array(x_band1)
# grad_band2 = np.array(x_band2)
# for idx, band in enumerate(grad_band1):
# 	grad = np.gradient(band)
# 	m_grad = np.sqrt(grad[0]**2 + grad[1]**2)
# 	grad_band1[idx] = m_grad
# for idx, band in enumerate(grad_band2):
# 	grad = np.gradient(band)
# 	m_grad = np.sqrt(grad[0]**2 + grad[1]**2)
# 	grad_band2[idx] = m_grad

filtered1 = np.load("bilateral1.npy")

print("done2")
y_train = np.array(train["is_iceberg"])
y_full = y_train.reshape(y_train.size,1)

# x_band1 = (x_band1-x_band1.min())/(x_band1.max()-x_band1.min())
# x_band2 = (x_band2-x_band2.min())/(x_band2.max()-x_band2.min())
ratio = (x_band1+x_band2)/2
X_full = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis], ratio[:,:,:,np.newaxis]], axis = -1)

print np.shape(x_band1)
print np.shape(x_band2)
print np.shape(X_full)
print np.shape(X_angle_train)
print np.shape(y_train)

X_train, X_valid, X_angle_train, X_angle_valid, y_train, y_valid = train_test_split(X_full,X_angle_train,y_full, train_size=0.85)


from matplotlib import pyplot
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Convolution2D, AveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.constraints import maxnorm
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.merge import Concatenate
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

def get_callbacks(filepath, patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]
    
def get_model():
    bn_model = 0
    p_activation = "relu"
    input_img = Input(shape=(75, 75, 3), name="X_1_lee")
    
    #did pretty well ~0.17 log loss
    # img_1 = Conv2D(128, kernel_size = (3,3),padding="same", activation=p_activation) ((BatchNormalization(momentum=bn_model))(input_1))
    # img_1 = MaxPooling2D((2,2)) (img_1)
    # img_1 = Dense(256, activation = 'relu') (img_1)
    # img_1 = Dropout(0.3)(img_1)
    # img_1 = Conv2D(64, kernel_size = (3,3),padding="same", activation=p_activation) (img_1)
    # img_1 = MaxPooling2D((2,2)) (img_1)
    # img_1 = Dense(128, activation = 'relu') (img_1)
    # img_1 = Dropout(0.3)(img_1)
    # img_1 = Conv2D(32, kernel_size = (3,3),padding="same", activation=p_activation) (img_1)
    # img_1 = MaxPooling2D((2,2)) (img_1)
    # img_1 = Dense(64, activation = 'relu') (img_1)
    # img_1 = Dropout(0.3)(img_1)

    # img_1 = GlobalMaxPooling2D() (img_1)
    conv1 = Convolution2D(
        96, (7, 7), activation='relu', kernel_initializer='glorot_uniform',
        strides=(2, 2), padding='same', name='conv1',
        data_format="channels_last")(input_img)
    print(conv1.shape)
    maxpool1 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool1',
        data_format="channels_last")(conv1)
    fire2_squeeze = Convolution2D(
        16, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_squeeze',
        data_format="channels_last")(maxpool1)
    fire2_expand1 = Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_expand1',
        data_format="channels_last")(fire2_squeeze)
    fire2_expand2 = Convolution2D(
        64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_expand2',
        data_format="channels_last")(fire2_squeeze)
    merge2 = Concatenate(axis=1)([fire2_expand1, fire2_expand2])

    fire3_squeeze = Convolution2D(
        16, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_squeeze',
        data_format="channels_last")(merge2)
    fire3_expand1 = Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_expand1',
        data_format="channels_last")(fire3_squeeze)
    fire3_expand2 = Convolution2D(
        64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_expand2',
        data_format="channels_last")(fire3_squeeze)
    merge3 = Concatenate(axis=1)([fire3_expand1, fire3_expand2])

    fire4_squeeze = Convolution2D(
        32, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_squeeze',
        data_format="channels_last")(merge3)
    fire4_expand1 = Convolution2D(
        128, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_expand1',
        data_format="channels_last")(fire4_squeeze)
    fire4_expand2 = Convolution2D(
        128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_expand2',
        data_format="channels_last")(fire4_squeeze)
    merge4 = Concatenate(axis=1)([fire4_expand1, fire4_expand2])
    maxpool4 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool4',
        data_format="channels_last")(merge4)

    fire5_squeeze = Convolution2D(
        32, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_squeeze',
        data_format="channels_last")(maxpool4)
    fire5_expand1 = Convolution2D(
        128, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_expand1',
        data_format="channels_last")(fire5_squeeze)
    fire5_expand2 = Convolution2D(
        128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_expand2',
        data_format="channels_last")(fire5_squeeze)
    merge5 = Concatenate(axis=1)([fire5_expand1, fire5_expand2])

    fire6_squeeze = Convolution2D(
        48, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_squeeze',
        data_format="channels_last")(merge5)
    fire6_expand1 = Convolution2D(
        192, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_expand1',
        data_format="channels_last")(fire6_squeeze)
    fire6_expand2 = Convolution2D(
        192, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_expand2',
        data_format="channels_last")(fire6_squeeze)
    merge6 = Concatenate(axis=1)([fire6_expand1, fire6_expand2])

    fire7_squeeze = Convolution2D(
        48, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire7_squeeze',
        data_format="channels_last")(merge6)
    fire7_expand1 = Convolution2D(
        192, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire7_expand1',
        data_format="channels_last")(fire7_squeeze)
    fire7_expand2 = Convolution2D(
        192, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire7_expand2',
        data_format="channels_last")(fire7_squeeze)
    merge7 = Concatenate(axis=1)([fire7_expand1, fire7_expand2])

    # fire8_squeeze = Convolution2D(
    #     64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
    #     padding='same', name='fire8_squeeze',
    #     data_format="channels_last")(merge7)
    # fire8_expand1 = Convolution2D(
    #     256, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
    #     padding='same', name='fire8_expand1',
    #     data_format="channels_last")(fire8_squeeze)
    # fire8_expand2 = Convolution2D(
    #     256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
    #     padding='same', name='fire8_expand2',
    #     data_format="channels_last")(fire8_squeeze)
    # merge8 = Concatenate(axis=1)([fire8_expand1, fire8_expand2])

    # maxpool8 = MaxPooling2D(
    #     pool_size=(3, 3), strides=(2, 2), name='maxpool8',
    #     data_format="channels_last")(merge8)
    # fire9_squeeze = Convolution2D(
    #     64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
    #     padding='same', name='fire9_squeeze',
    #     data_format="channels_last")(maxpool8)
    # fire9_expand1 = Convolution2D(
    #     256, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
    #     padding='same', name='fire9_expand1',
    #     data_format="channels_last")(fire9_squeeze)
    # fire9_expand2 = Convolution2D(
    #     256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
    #     padding='same', name='fire9_expand2',
    #     data_format="channels_last")(fire9_squeeze)
    # merge9 = Concatenate(axis=1)([fire9_expand1, fire9_expand2])

    fire9_dropout = Dropout(0.5, name='fire9_dropout')(merge7)
    conv10 = Convolution2D(
        1, (1, 1), kernel_initializer='glorot_uniform',
        padding='valid', name='conv10',
        data_format="channels_last")(fire9_dropout)
    # The size should match the output of conv10
    avgpool10 = AveragePooling2D(
        (4, 4), name='avgpool10',
        data_format="channels_last")(conv10)

    flatten = Flatten(name='flatten')(avgpool10)

    output = Dense(1, activation="sigmoid")(flatten)
    
    model = Model(input_img,  output)
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

model = get_model()
print(model.summary())

file_path = ".model_weights.hdf5"
callbacks = get_callbacks(filepath=file_path, patience=100)

model.fit(X_train, y_train, epochs=50
          , validation_data=(X_valid, y_valid)
         , batch_size=32
         , callbacks=callbacks)
# combo_gen = map(format_gen_outputs, datagen.flow(X_train[:,:,:,],y_train, batch_size=32),
                # datagen.flow(X_train[:,:,:,2:4],y_train, batch_size=32))
# model.fit_generator(datagen.flow(X_train,y_train, batch_size=32), 
#                     validation_data=(X_valid, y_valid),
#                     callbacks = callbacks,
#                     steps_per_epoch=len(X_train)/16, epochs=50)

# estimator = KerasClassifier(build_fn=get_model,nb_epoch=25,batch_size=5,verbose=0)
# kfold=StratifiedKFold(n_splits=5,shuffle=True, random_state=123)
# cvscores = []
# for train, test in kfold.split(X_full, y_full):
#     # Fit the model
#     model.fit([X_full[train][:,:,:,:2],X_full[train][:,:,:,2:4]], y_full[train],epochs=25, batch_size=32)

#     scores = model.evaluate([X_full[test][:,:,:,:2],X_full[test][:,:,:,2:4]], y_full[test], verbose=0)
#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#     cvscores.append(scores[1] * 100)

# print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
# results = cross_val_score(estimator,X_full,y_full, cv=kfold)
# print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

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