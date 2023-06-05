import modules as md
import models
import tensorflow as tf

Input = tf.keras.layers.Input(shape=(64, 64, 3))
layers = models.build_mondi_model(input_layers=Input)


model = tf.keras.models.Model(Input, layers[-1])
print(model.summary())
