import modules as md
import models
import tensorflow as tf

Input = tf.keras.layers.Input(shape=(64, 64, 3))
x = md.channel_attention_module(Input, 64)
# layers = models.build_mondi_model(input_layers=Input)


model = tf.keras.models.Model(Input, x)
print(model.summary())
