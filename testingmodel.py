import modules as md
import metrics

import tensorflow as tf
import numpy as np

x_test = np.load(f"np_data/2.npy")
y_test = np.load(f"np_data/3.npy")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Input = tf.keras.layers.Input(shape=x_test.shape[1:])
# layers = models.build_mondi_model(input_layers=Input)


# model = tf.keras.models.Model(Input, x)
rmse_metrics = tf.keras.metrics.RootMeanSquaredError()
r2 =  metrics.R_squared

model = tf.keras.models.load_model('fc_mondi_1_regression.h5', compile=False)
# model.compile(optimizer='adam',
#               loss = 'mse',
#               metrics=['mae', rmse_metrics, metrics.R_squared])

print(model.summary())
model.predict(x_test)