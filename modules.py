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
  pool = input.get_shape()[1:-1]
  avgpool = tf.keras.layers.AvgPool2D(pool)(input)
  maxpool = tf.keras.layers.MaxPool2D(pool)(input)
  
  mlp1 = tf.keras.layers.Dense(filter//ratio)(avgpool)
  mlp1 = tf.keras.layers.Dense(filter)(mlp1)

  mlp2 = tf.keras.layers.Dense(filter//ratio)(maxpool)
  mlp2 = tf.keras.layers.Dense(filter)(mlp2)

  mlp = mlp1 + mlp2 # tf.keras.layers.Add(activation='relu)([mlp1, mlp2])
  mlp = tf.nn.relu(mlp)
  return input * mlp # tf.keras.layers.Multiply()([input, mlp])

def spatial_attention_module(input, kernel_size=7):
  pool = input.get_shape()[1:-1]
  avgpool = tf.keras.layers.AvgPool2D(pool)(input)
  maxpool = tf.keras.layers.MaxPool2D(pool)(input)
  x = Concat([avgpool, maxpool])
  x = tf.keras.layers.Conv2D(1, padding='same', use_bias=False,
                             kernel_size=kernel_size,
                             #activation='sigmoid'
                             )(x)
  # x = tf.keras.layers.Concatenate()([x, x])
  
  return input * x # tf.keras.layers.Multiply()([input, x])

def cbam_block(input, filter):
  cam = channel_attention_module(input, filter=filter)
  sam = spatial_attention_module(cam)
  return sam

# def cbam_block(input_feature, name='cbam', ratio=8):
#   """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
#   As described in https://arxiv.org/abs/1807.06521.
#   """
  
#   # with tf.variable_scope(name):
#   attention_feature = channel_attention(input_feature, ratio)# 'ch_at', ratio)
#   attention_feature = spatial_attention(attention_feature) # 'sp_at')
#   # print ("CBAM Hello")
#   return attention_feature

# def channel_attention(input_feature, ratio=8):
  
#   # kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
#   # bias_initializer = tf.constant_initializer(value=0.0)
  
#   # with tf.variable_scope(name):
    
#   channel = input_feature.get_shape()[-1]
#   avg_pool = tf.reduce_mean(input_feature, axis=[1,2], keepdims=True)
      
#   assert avg_pool.get_shape()[1:] == (1,1,channel)
#   avg_pool = tf.keras.layers.Dense(# inputs=avg_pool,
#                                 units=channel//ratio,
#                                 activation=tf.nn.relu,
#                                 # kernel_initializer=kernel_initializer,
#                                 # bias_initializer=bias_initializer,
#                                 name='mlp_0',
#                                 # reuse=None,
#                                 )   
#   assert avg_pool.get_shape()[1:] == (1,1,channel//ratio)
#   avg_pool = tf.layers.Dense(# inputs=avg_pool,
#                                 units=channel,                             
#                                 # kernel_initializer=kernel_initializer,
#                                 # bias_initializer=bias_initializer,
#                                 name='mlp_1',
#                                 # reuse=None,
#                                 )    
#   assert avg_pool.get_shape()[1:] == (1,1,channel)

#   max_pool = tf.reduce_max(input_feature, axis=[1,2], keepdims=True)    
#   assert max_pool.get_shape()[1:] == (1,1,channel)
#   max_pool = tf.layers.Dense(# inputs=max_pool,
#                                 units=channel//ratio,
#                                 activation=tf.nn.relu,
#                                 name='mlp_0',
#                                 # reuse=True
#                                 )   
#   assert max_pool.get_shape()[1:] == (1,1,channel//ratio)
#   max_pool = tf.layers.Dense(# inputs=max_pool,
#                                 units=channel,                             
#                                 name='mlp_1',
#                                 # reuse=True
#                                 )  
#   assert max_pool.get_shape()[1:] == (1,1,channel)

#   scale = tf.sigmoid(avg_pool + max_pool, 'sigmoid')
    
#   return input_feature * scale

# def spatial_attention(input_feature):
#   kernel_size = 7
#   # kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
#   # with tf.variable_scope(name):
#   avg_pool = tf.reduce_mean(input_feature, axis=[3], keepdims=True)
#   assert avg_pool.get_shape()[-1] == 1
#   max_pool = tf.reduce_max(input_feature, axis=[3], keepdims=True)
#   assert max_pool.get_shape()[-1] == 1
#   concat = tf.concat([avg_pool,max_pool], 3)
#   assert concat.get_shape()[-1] == 2
  
#   concat = tf.keras.layers.Conv2D(concat,
#                             filters=1,
#                             kernel_size=[kernel_size,kernel_size],
#                             strides=[1,1],
#                             padding="same",
#                             activation=None,
#                             # kernel_initializer=kernel_initializer,
#                             use_bias=False,
#                             name='conv')
#   assert concat.get_shape()[-1] == 1
#   concat = tf.sigmoid(concat, 'sigmoid')
    
#   return input_feature * concat

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
  x = Dense(x, 32)
  x = Dense(x, 16)
  x = Dense(x, 1)
  return x

def RegFCMondi(input, fc=False):
  x = Dense(input, 64)
  x = Flatten(x)
  
  if fc:
    x = Dense(input, 128)
    x = Dense(x, 64)
    x = Flatten(x)
    x = Dense(x, 64)

  # x = Dense(x, 64)
  x = Dense(x, 16)
  x = Dense(x, 4)
  x = Dense(x, 1)
  return x

def RegFlat(input):
  x = Flatten(input)
  x = Dense(x, 1)
  return x