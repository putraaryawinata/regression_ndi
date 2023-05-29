import tensorflow as tf

def Dense(input, units, activation='relu'):
    x = tf.keras.layers.Dense(units, activation=activation)(input)
    return x

def Flatten(input):
    return tf.keras.layers.Flatten()(input)

def Conv(input, filter, kernel_size, stride=1, padding='same', activation=tf.nn.swish):
  x = tf.keras.layers.Conv2D(filter, kernel_size, padding=padding, strides=stride,
                            use_bias=False, activation=activation)(input)
  x = tf.keras.layers.BatchNormalization()(x)
  return x

def Concat(list_layer, axis=-1):
  return tf.keras.layers.Concatenate(axis=axis)(list_layer)

def MP(input, kernel_size=2, stride=2, padding='same'):
  return tf.keras.layers.MaxPool2D(kernel_size, strides=stride, padding=padding)(input)

def SPPCSPC(input, filter):
  x1, x2 = Conv(input, filter, kernel_size=1), Conv(input, filter, kernel_size=1)
  x1 = Conv(x1, filter, kernel_size=3)
  x1 = Conv(x1, filter, kernel_size=1)
  x11 = MP(x1, kernel_size=5, stride=1)
  x12 = MP(x1, kernel_size=9, stride=1)
  x13 = MP(x1, kernel_size=13, stride=1)
  x1 = Concat([x1, x11, x12, x13])
  x1 = Conv(x1, filter, kernel_size=1)
  x1 = Conv(x1, filter, kernel_size=3)
  x = Concat([x1, x2])
  x = Conv(x, filter // 2, kernel_size=1)
  return x

def Upsample(input, size=2):
  return tf.keras.layers.UpSampling2D(size=size)(input)

def RepConv(input, filter, kernel_size=3, stride=1, name='repconv'):
  x1 = Conv(input, filter, kernel_size, activation=None)
  x2 = Conv(input, filter, 1, padding='valid', activation=None)
  if input.shape[-1] == filter:
    x3 = tf.keras.layers.BatchNormalization()(input)
    x = tf.keras.layers.Add(name=name)([x1, x2, x3])
  else:
    x = tf.keras.layers.Add(name=name)([x1, x2])
  # x = Conv(x, filter, kernel_size=1)
  return x

def Last(list_input, num_classes, name='bbox'):
  x = []
  for i in range(len(list_input)):
    x.append(Conv(list_input[i], 3 * (num_classes + 5), 1, activation=None))
  return x

# Regression Modules
def RegFC(input):
  x = Flatten(input)
  x = Dense(x, 1024)
  x = Dense(x, 256)
  x = Dense(x, 64)
  x = Dense(x, 16)
  x = Dense(x, 1)
  return x

def RegFlat(input):
  x = Flatten(input)
  x = Dense(x, 1)
  return x