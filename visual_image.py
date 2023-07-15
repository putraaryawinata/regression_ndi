import tensorflow as tf
import modules as md

def la(x, xi): # layers addition
  return x.append(xi)

input_layer = tf.keras.layers.Input((64, 64, 1))

def mondi_regression(input_layers, mode='auto', filename='auto_yolo'):
  x = []
  # Backbone
  la(x, md.GhostConv(input_layers, 32, 3, 1)) # 0

  la(x, md.GhostConv(x[-1], 32, 3, 1)) # 1
  la(x, md.GhostConv(x[-1], 64, 3, 1))

  la(x, md.GhostConv(x[-1], 128, 3, 2)) # 3-P2/4
  if mode=='fc':
    la(x, md.Flatten(x[-1]))
    la(x, md.Dense(x[-1], 16))

  if mode=='cnn':
    la(x, tf.keras.layers.Conv2D(128, 3, activation='relu')(x[-1]))
    la(x, tf.keras.layers.MaxPool2D()(x[-1]))
    la(x, tf.keras.layers.Conv2D(256, 3, activation='relu')(x[-1]))
    la(x, tf.keras.layers.MaxPool2D()(x[-1]))
    la(x, tf.keras.layers.Conv2D(512, 3, activation='relu')(x[-1]))
    la(x, md.Flatten(x[-1]))

  if mode=='auto':
    la(x, tf.keras.layers.Conv2D(128, 3, activation='relu')(x[-1])) # encoder
    la(x, tf.keras.layers.MaxPool2D()(x[-1]))
    la(x, tf.keras.layers.Conv2D(256, 3, activation='relu')(x[-1]))
    la(x, tf.keras.layers.MaxPool2D()(x[-1]))
    la(x, tf.keras.layers.Conv2D(512, 3, activation='relu')(x[-1]))
    la(x, tf.keras.layers.Conv2D(512, 3, activation='relu')(x[-1])) # decoder
    la(x, tf.keras.layers.UpSampling2D()(x[-1]))
    la(x, tf.keras.layers.Conv2D(256, 3, activation='relu')(x[-1]))
    la(x, tf.keras.layers.UpSampling2D()(x[-1]))
    la(x, tf.keras.layers.Conv2D(128, 3, activation='relu')(x[-1]))
    la(x, md.Flatten(x[-1]))
  
  la(x, md.Dense(x[-1], 4))
  la(x, md.Dense(x[-1], 1))

model.summary()