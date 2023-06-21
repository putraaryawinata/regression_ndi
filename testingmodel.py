import modules as md
import models
import tensorflow as tf

Input = tf.keras.layers.Input(shape=(64, 64, 3))
# layers = models.build_mondi_model(input_layers=Input)


# model = tf.keras.models.Model(Input, x)
model = tf.keras.models.load_model('auto_mondi_1_regression.h5', compile=False)
print(model.summary())
