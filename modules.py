import tensorflow as tf

def Dense(input, units, activation='relu'):
    x = tf.keras.layers.Dense(units, activation=activation)(input)
    return x

def Flatten(input):
    return tf.keras.layers.Flatten()(input)

def Conv(input, filter, kernel_size, stride=1, groups=1, padding='same', activation=tf.nn.swish):
  x = tf.keras.layers.Conv2D(filter, kernel_size, padding=padding, strides=stride,
                             groups=groups, use_bias=False, activation=activation)(input)
  x = tf.keras.layers.BatchNormalization()(x)
  return x

def GhostConv(input, filter, kernel_size=1, stride=1, ratio=2, dw_size=3, act='relu'):
  init_channel = (filter // ratio) + 1
  new_channel = init_channel * (ratio-1)
  primary_conv = Conv(input, init_channel, kernel_size, stride, activation=act)
  cheap_operation = Conv(primary_conv, new_channel, dw_size, 1, groups=init_channel, activation=act)
  out = Concat([primary_conv, cheap_operation])
  return out[:,:,:,:filter]
  # return out

def channel_attention_module(input, filter, ratio=8):
  avgpool = tf.keras.layers.GlobalAveragePooling2D()(input)
  maxpool = tf.keras.layers.GlobalMaxPooling2D()(input)
  
  mlp1 = tf.keras.layers.Dense(filter//ratio, activation='relu')(avgpool)
  mlp1 = tf.keras.layers.Dense(filter)(mlp1)

  mlp2 = tf.keras.layers.Dense(filter//ratio, activation='relu')(maxpool)
  mlp2 = tf.keras.layers.Dense(filter)(mlp2)

  mlp = tf.keras.layers.Add()([mlp1, mlp2])
  mlp = tf.expand_dims(mlp)
  mlp = tf.expand_dims(mlp)
  return input * mlp

def spatial_attention_module(input, kernel_size=7):
  avgpool = tf.keras.layers.GlobalAveragePooling2D()(input)
  maxpool = tf.keras.layers.GlobalMaxPooling2D()(input)
  x = Concat([avgpool, maxpool])
  x = tf.keras.layers.Conv2D(1, kernel_size=kernel_size,
                             padding='same', use_bias=False,
                             activation='sigmoid')(x)
  return tf.keras.layers.Multiply()([input, x])

def CBAM(input, filter):
  cam = channel_attention_module(input, filter=filter)
  sam = spatial_attention_module(cam)
  return sam

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