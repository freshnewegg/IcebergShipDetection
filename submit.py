import pandas as pd
from pandas import read_csv
import numpy as np
from pyradar.filters.frost import frost_filter
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance

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
X_angle_test = np.array(test.inc_angle)

for idx, band in enumerate(x_band1):
	band = lee_filter(np.multiply(10**(band/20), np.sin(X_angle_test[idx]*3.14159/180)/np.cos(X_angle_test[idx]*3.14159/180)),10)
for idx, band in enumerate(x_band2):
	band = lee_filter(np.multiply(10**(band/20), np.sin(X_angle_test[idx]*3.14159/180)/np.cos(X_angle_test[idx]*3.14159/180)),10)

X_test = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis]], axis = -1)

model = load_model('.model_weights.hdf5')

prediction = model.predict(X_test, verbose=1, batch_size=200)

submission = pd.DataFrame({'id': test["id"], 'is_iceberg': prediction.reshape((prediction.shape[0]))})

submission.to_csv("./submission.csv", index=False)