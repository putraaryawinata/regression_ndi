import tensorflow as tf
import numpy as np

import models, modules, preprocess, metrics
import callbacks as cb

# x_train, y_train = preprocess.load(path="resized_dataset/train/images/*.bmp")
# x_valid, y_valid = preprocess.load(path="resized_dataset/valid/images/*.bmp")

# x_train, y_train, x_valid, y_valid = preprocess.expand([x_train, y_train, x_valid, y_valid])
# preprocess.save([x_train, y_train, x_valid, y_valid], "np_data")

x_train = np.load(f"np_data/0.npy")
y_train = np.load(f"np_data/1.npy")
x_valid = np.load(f"np_data/2.npy")
y_valid = np.load(f"np_data/3.npy")

print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_valid shape: {x_valid.shape}")
print(f"y_valid shape: {y_valid.shape}")

Input = tf.keras.layers.Input(shape=x_train.shape[1:])
layers = models.build_mondi_model(input_layers=Input)

model = tf.keras.models.Model(Input, layers[-1])

rmse_metrics = tf.keras.metrics.RootMeanSquaredError()
# r2 =  metrics.RSquare()

model.compile(optimizer='adam',
              loss = 'mse',
              metrics=['mae', rmse_metrics, metrics.R_squared])

print(model.summary())
# saved_best_ckpt = cb.best_ckpt("mondi_fc_cbam")

# history = model.fit(x_train, y_train, batch_size=16, epochs=500,
#                     validation_data=(x_valid, y_valid),
#                     callbacks=[cb.early_stopping, saved_best_ckpt])

# metrics.dict_to_json(history.history, file_name="fc_mondi_cbam")
# model.save('fc_mondi_cbam_regression.h5')