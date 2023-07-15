# TO GAIN IMAGES AS THE RESULT OF THE AUTOENCODER
import glob
import re
import os

import metrics
import callbacks as cb

import numpy as np
from PIL import Image
import tensorflow as tf

def np_to_img(nparray, path, namefile):
    img = Image.fromarray(nparray)
    img_name = os.path.join(path, namefile)
    img.save(img_name)
    return img

def img_to_np(filename):
    im = Image.open(filename)
    arr = np.asarray(im)
    return arr

def angle_correction(angle):
    angle = 90 - abs(angle%180 - 90)
    return angle

## Data Preprocessing
train_images = glob.glob("resized_dataset/train/images/*.bmp")
test_images = glob.glob("resized_dataset/test/images/*.bmp")

x_train = []
x_test = []
y_train = []
y_test = []

for i, img in enumerate(train_images):
    arr = img_to_np(img)
    x_train.append(arr)
    data = re.search("images\/([A-Z1-9]*)_(\d.\d*)_(\d*)", img)
    _, _, angle = data.group(1), float(data.group(2)), int(data.group(3))
    angle = angle_correction(angle)
    y_train.append(angle)

for i, img in enumerate(test_images):
    arr = img_to_np(img)
    x_train.append(arr)
    data = re.search("images\/([A-Z1-9]*)_(\d.\d*)_(\d*)", img)
    _, _, angle = data.group(1), float(data.group(2)), int(data.group(3))
    angle = angle_correction(angle)
    y_train.append(angle)

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

## Create Model
latent_dim = 512 

class Autoencoder(tf.keras.models.Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim   
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(latent_dim//4, 3, activation='relu'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(latent_dim//2, 3, activation='relu'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(latent_dim, 3, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(latent_dim, 3, activation='relu'), # decoder
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(latent_dim//2, 3, activation='relu'), # decoder
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(latent_dim//4, 3, activation='relu'), # decoder
        ])
        self.fullyconnected = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        output = self.fullyconnected(decoded)
        return output

autoencoder = Autoencoder(latent_dim)

## COMPILE AND TRAIN
rmse_metrics = tf.keras.metrics.RootMeanSquaredError()
autoencoder.compile(optimizer="adam",
                    loss="mse",
                    metrics=['mae', rmse_metrics, metrics.R_squared])

autoencoder.fit(x_train, y_train,
                validation_data=(x_test, y_test),
                batch_size=16, epochs=500,
                callbacks=[cb.early_stopping])

autoencoder.save("visual.h5")

## LOAD MODEL
autoencoder = tf.keras.models.load_model("visual.h5", compile=False)
rmse_metrics = tf.keras.metrics.RootMeanSquaredError()
autoencoder.compile(optimizer="adam",
                    loss="mse",
                    metrics=['mae', rmse_metrics, metrics.R_squared])

encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()