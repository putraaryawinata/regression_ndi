import tensorflow as tf
import modules as md

# def GhostConv():
#   return 1

def la(x, xi): # layers addition
  return x.append(xi)

def build_custom_model(input_layers, num_classes=13):
  x = []
  la(x, md.GhostConv(input_layers, 32, 1))
  la(x, md.GhostConv(x[-1], 32, 1))
  la(x, md.Flatten(x[-1]))
  la(x, md.Dense(x[-1], num_classes))
  return x

def build_yolov7_model(input_layers, num_classes=13):
  x = []
  # Backbone
  la(x, md.Conv(input_layers, 32, 3, 1)) # 0

  la(x, md.Conv(x[-1], 32, 3, 1)) # 1
  la(x, md.Conv(x[-1], 64, 3, 1))

  la(x, md.Conv(x[-1], 128, 3, 2)) # 3-P2/4
  la(x, md.Conv(x[-1], 64, 1, 1))
  la(x, md.Conv(x[-2], 64, 1, 1))
  la(x, md.Conv(x[-1], 64, 3, 1))
  la(x, md.Conv(x[-1], 64, 3, 1))
  la(x, md.Conv(x[-1], 64, 3, 1))
  la(x, md.Conv(x[-1], 64, 3, 1))
  la(x, md.Concat([x[-1], x[-3], x[-5], x[-6]]))
  la(x, md.Conv(x[-1], 256, 1, 1)) # 11

  la(x, md.MP(x[-1]))
  la(x, md.Conv(x[-1], 128, 1, 1))
  la(x, md.Conv(x[-3], 128, 1, 1))
  la(x, md.Conv(x[-1], 128, 3, 2))
  la(x, md.Concat([x[-1], x[-3]])) # 16-P3/8
  la(x, md.Conv(x[-1], 128, 1, 1))
  la(x, md.Conv(x[-2], 128, 1, 1))
  la(x, md.Conv(x[-1], 128, 3, 1))
  la(x, md.Conv(x[-1], 128, 3, 1))
  la(x, md.Conv(x[-1], 128, 3, 1))
  la(x, md.Conv(x[-1], 128, 3, 1))
  la(x, md.Concat([x[-1], x[-3], x[-5], x[-6]]))
  la(x, md.Conv(x[-1], 512, 1, 1)) # 24

  la(x, md.MP(x[-1]))
  la(x, md.Conv(x[-1], 256, 1, 1))
  la(x, md.Conv(x[-3], 256, 1, 1))
  la(x, md.Conv(x[-1], 256, 3, 2))
  la(x, md.Concat([x[-1], x[-3]])) # 29-P4/16
  la(x, md.Conv(x[-1], 256, 1, 1))
  la(x, md.Conv(x[-2], 256, 1, 1))
  la(x, md.Conv(x[-1], 256, 3, 1))
  la(x, md.Conv(x[-1], 256, 3, 1))
  la(x, md.Conv(x[-1], 256, 3, 1))
  la(x, md.Conv(x[-1], 256, 3, 1))
  la(x, md.Concat([x[-1], x[-3], x[-5], x[-6]]))
  la(x, md.Conv(x[-1], 1024, 1, 1)) # 37

  la(x, md.MP(x[-1]))  #  [-1, 1, MP, []],
  la(x, md.Conv(x[-1], 512, 1, 1))
  la(x, md.Conv(x[-3], 512, 1, 1))
  la(x, md.Conv(x[-1], 512, 3, 2))
  la(x, md.Concat([x[-1], x[-3]])) # 42-P5/32
  la(x, md.Conv(x[-1], 256, 1, 1))
  la(x, md.Conv(x[-2], 256, 1, 1))
  la(x, md.Conv(x[-1], 256, 3, 1))
  la(x, md.Conv(x[-1], 256, 3, 1))
  la(x, md.Conv(x[-1], 256, 3, 1))
  la(x, md.Conv(x[-1], 256, 3, 1))
  la(x, md.Concat([x[-1], x[-3], x[-5], x[-6]]))
  la(x, md.Conv(x[-1], 1024, 1, 1)) # 50

  # Head
  la(x, md.SPPCSPC(x[50], 512))  # [[-1, 1, SPPCSPC, [512]], # 51

  la(x, md.Conv(x[-1], 256, 1, 1))  #  [-1, 1, Conv, [256, 1, 1]],
  la(x, md.Upsample(x[-1], 2))  #  [-1, 1, nn.Upsample, [None, 2, 'nearest']],
  la(x, md.Conv(x[37], 256, 1, 1))  #  [37, 1, Conv, [256, 1, 1]], # route backbone P4
  la(x, md.Concat([x[-1], x[-2]]))  #  [[-1, -2], 1, Concat, [1]],

  la(x, md.Conv(x[-1], 256, 1, 1))  #  [-1, 1, Conv, [256, 1, 1]],
  la(x, md.Conv(x[-2], 256, 1, 1))  #  [-2, 1, Conv, [256, 1, 1]],
  la(x, md.Conv(x[-1], 128, 3, 1))  #  [-1, 1, Conv, [128, 3, 1]],
  la(x, md.Conv(x[-1], 128, 3, 1))  #  [-1, 1, Conv, [128, 3, 1]],
  la(x, md.Conv(x[-1], 128, 3, 1))  #  [-1, 1, Conv, [128, 3, 1]],
  la(x, md.Conv(x[-1], 128, 3, 1))  #  [-1, 1, Conv, [128, 3, 1]],
  la(x, md.Concat([x[-1], x[-2], x[-3], x[-4], x[-5], x[-6]], -1))  #  [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
  la(x, md.Conv(x[-1], 256, 1, 1))  #  [-1, 1, Conv, [256, 1, 1]], # 63

  la(x, md.Conv(x[-1], 128, 1, 1))  #  [-1, 1, Conv, [128, 1, 1]],
  la(x, md.Upsample(x[-1], 2))  #  [-1, 1, nn.Upsample, [None, 2, 'nearest']],
  la(x, md.Conv(x[24], 128, 1, 1)) #  [24, 1, Conv, [128, 1, 1]], # route backbone P3
  la(x, md.Concat([x[-1], x[-2]]))  #  [[-1, -2], 1, Concat, [1]],

  la(x, md.Conv(x[-1], 128, 1, 1))  #  [-1, 1, Conv, [128, 1, 1]],
  la(x, md.Conv(x[-2], 128, 1, 1))  #  [-2, 1, Conv, [128, 1, 1]],
  la(x, md.Conv(x[-1], 64, 3, 1))  #  [-1, 1, Conv, [64, 3, 1]],
  la(x, md.Conv(x[-1], 64, 3, 1))  #  [-1, 1, Conv, [64, 3, 1]],
  la(x, md.Conv(x[-1], 64, 3, 1))  #  [-1, 1, Conv, [64, 3, 1]],
  la(x, md.Conv(x[-1], 64, 3, 1))  #  [-1, 1, Conv, [64, 3, 1]],
  la(x, md.Concat([x[-1], x[-2], x[-3], x[-4], x[-5], x[-6]]))  #  [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
  la(x, md.Conv(x[-1], 128, 1, 1))  #  [-1, 1, Conv, [128, 1, 1]], # 75

  # la(x, md.MP(x[-1]))  #  [-1, 1, MP, []],
  # la(x, md.Conv(x[-1], 128, 1, 1))  #  [-1, 1, Conv, [128, 1, 1]],
  # la(x, md.Conv(x[-3], 128, 1, 1))  #  [-3, 1, Conv, [128, 1, 1]],
  # la(x, md.Conv(x[-1], 128, 3, 2))  #  [-1, 1, Conv, [128, 3, 2]],
  # la(x, md.Concat([x[-1], x[-3], x[63]]))  #  [[-1, -3, 63], 1, Concat, [1]],

  # la(x, md.Conv(x[-1], 256, 1, 1))  #  [-1, 1, Conv, [256, 1, 1]],
  # la(x, md.Conv(x[-2], 256, 1, 1))  #  [-2, 1, Conv, [256, 1, 1]],
  # la(x, md.Conv(x[-1], 128, 3, 1))  #  [-1, 1, Conv, [128, 3, 1]],
  # la(x, md.Conv(x[-1], 128, 3, 1))  #  [-1, 1, Conv, [128, 3, 1]],
  # la(x, md.Conv(x[-1], 128, 3, 1))  #  [-1, 1, Conv, [128, 3, 1]],
  # la(x, md.Conv(x[-1], 128, 3, 1))  #  [-1, 1, Conv, [128, 3, 1]],
  # la(x, md.Concat([x[-1], x[-2], x[-3], x[-4], x[-5], x[-6]]))  #  [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
  # la(x, md.Conv(x[-1], 256, 1, 1))  #  [-1, 1, Conv, [256, 1, 1]], # 88

  # la(x, md.MP(x[-1]))  #  [-1, 1, MP, []],
  # la(x, md.Conv(x[-1], 256, 1, 1))  #  [-1, 1, Conv, [256, 1, 1]],
  # la(x, md.Conv(x[-3], 256, 1, 1))  #  [-3, 1, Conv, [256, 1, 1]],
  # la(x, md.Conv(x[-1], 256, 3, 2))  #  [-1, 1, Conv, [256, 3, 2]],
  # la(x, md.Concat([x[-1], x[-3], x[51]]))  #  [[-1, -3, 51], 1, Concat, [1]],

  # la(x, md.Conv(x[-1], 512, 1, 1))  #  [-1, 1, Conv, [512, 1, 1]],
  # la(x, md.Conv(x[-2], 512, 1, 1))  #  [-2, 1, Conv, [512, 1, 1]],
  # la(x, md.Conv(x[-1], 256, 3, 1))  #  [-1, 1, Conv, [256, 3, 1]],
  # la(x, md.Conv(x[-1], 256, 3, 1))  #  [-1, 1, Conv, [256, 3, 1]],
  # la(x, md.Conv(x[-1], 256, 3, 1))  #  [-1, 1, Conv, [256, 3, 1]],
  # la(x, md.Conv(x[-1], 256, 3, 1))  #  [-1, 1, Conv, [256, 3, 1]],
  # la(x, md.Concat([x[-1], x[-2], x[-3], x[-4], x[-5], x[-6]]))  #  [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
  # la(x, md.Conv(x[-1], 512, 1, 1))  #  [-1, 1, Conv, [512, 1, 1]], # 101

  # la(x, md.RepConv(x[75], 256, 3, 1, name='repconv1'))  #  [75, 1, RepConv, [256, 3, 1]], # 102
  # la(x, md.RepConv(x[88], 512, 3, 1, name='repconv2'))  #  [88, 1, RepConv, [512, 3, 1]], # 103
  # la(x, md.RepConv(x[101], 1024, 3, 1, name='repconv3'))  #  [101, 1, RepConv, [1024, 3, 1]], # 104


  # la(x, md.Last([x[102], x[103], x[104]], num_classes))

    #  [[102,103,104], 1, IDetect, [nc, anchors]],   # Detect(P3, P4, P5)
  # regression part
  la(x, md.RegFC(x[75]))

  # la(x, md.RegFlat(x[50]))
  
  # la(x, md.RegFlat(x[75]))

  return x

def build_mondi_model(input_layers, num_classes=13):
  x = []
  # Backbone
  la(x, md.GhostConv(input_layers, 32, 3, 1)) # 0

  la(x, md.GhostConv(x[-1], 32, 3, 1)) # 1
  la(x, md.GhostConv(x[-1], 64, 3, 1))

  la(x, md.GhostConv(x[-1], 128, 3, 2)) # 3-P2/4
  la(x, md.GhostConv(x[-1], 64, 1, 1))
  la(x, md.GhostConv(x[-2], 64, 1, 1))
  la(x, md.GhostConv(x[-1], 64, 3, 1))
  la(x, md.GhostConv(x[-1], 64, 3, 1))
  la(x, md.GhostConv(x[-1], 64, 3, 1))
  la(x, md.GhostConv(x[-1], 64, 3, 1))
  la(x, md.Concat([x[-1], x[-3], x[-5], x[-6]]))
  la(x, md.GhostConv(x[-1], 256, 1, 1)) # 11

  la(x, md.CBAM(x[-1], 256))

  la(x, md.MP(x[-1]))
  la(x, md.GhostConv(x[-1], 128, 1, 1))
  la(x, md.GhostConv(x[-3], 128, 1, 1))
  la(x, md.GhostConv(x[-1], 128, 3, 2))
  la(x, md.Concat([x[-1], x[-3]])) # 17-P3/8
  la(x, md.GhostConv(x[-1], 128, 1, 1))
  la(x, md.GhostConv(x[-2], 128, 1, 1))
  la(x, md.GhostConv(x[-1], 128, 3, 1))
  la(x, md.GhostConv(x[-1], 128, 3, 1))
  la(x, md.GhostConv(x[-1], 128, 3, 1))
  la(x, md.GhostConv(x[-1], 128, 3, 1))
  la(x, md.Concat([x[-1], x[-3], x[-5], x[-6]]))
  la(x, md.GhostConv(x[-1], 512, 1, 1)) # 25

  la(x, md.CBAM(x[-1], 512))

  la(x, md.MP(x[-1]))
  la(x, md.GhostConv(x[-1], 256, 1, 1))
  la(x, md.GhostConv(x[-3], 256, 1, 1))
  la(x, md.GhostConv(x[-1], 256, 3, 2))
  la(x, md.Concat([x[-1], x[-3]])) # 31-P4/16
  la(x, md.GhostConv(x[-1], 256, 1, 1))
  la(x, md.GhostConv(x[-2], 256, 1, 1))
  la(x, md.GhostConv(x[-1], 256, 3, 1))
  la(x, md.GhostConv(x[-1], 256, 3, 1))
  la(x, md.GhostConv(x[-1], 256, 3, 1))
  la(x, md.GhostConv(x[-1], 256, 3, 1))
  la(x, md.Concat([x[-1], x[-3], x[-5], x[-6]]))
  la(x, md.GhostConv(x[-1], 1024, 1, 1)) # 39

  la(x, md.CBAM(x[-1], 1024))

  la(x, md.MP(x[-1]))  #  [-1, 1, MP, []],
  la(x, md.GhostConv(x[-1], 512, 1, 1))
  la(x, md.GhostConv(x[-3], 512, 1, 1))
  la(x, md.GhostConv(x[-1], 512, 3, 2))
  la(x, md.Concat([x[-1], x[-3]])) # 45-P5/32
  la(x, md.GhostConv(x[-1], 256, 1, 1))
  la(x, md.GhostConv(x[-2], 256, 1, 1))
  la(x, md.GhostConv(x[-1], 256, 3, 1))
  la(x, md.GhostConv(x[-1], 256, 3, 1))
  la(x, md.GhostConv(x[-1], 256, 3, 1))
  la(x, md.GhostConv(x[-1], 256, 3, 1))
  la(x, md.Concat([x[-1], x[-3], x[-5], x[-6]]))
  la(x, md.GhostConv(x[-1], 1024, 1, 1)) # 53

  # Head
  la(x, md.SPPCSPC(x[53], 512))  # [[-1, 1, SPPCSPC, [512]], # 54

  la(x, md.GhostConv(x[-1], 256, 1, 1))  #  [-1, 1, Conv, [256, 1, 1]],
  la(x, md.Upsample(x[-1], 2))  #  [-1, 1, nn.Upsample, [None, 2, 'nearest']],
  la(x, md.GhostConv(x[39], 256, 1, 1))  #  [37, 1, Conv, [256, 1, 1]], # route backbone P4
  la(x, md.Concat([x[-1], x[-2]]))  #  [[-1, -2], 1, Concat, [1]],

  la(x, md.GhostConv(x[-1], 256, 1, 1))  #  [-1, 1, Conv, [256, 1, 1]],
  la(x, md.GhostConv(x[-2], 256, 1, 1))  #  [-2, 1, Conv, [256, 1, 1]],
  la(x, md.GhostConv(x[-1], 128, 3, 1))  #  [-1, 1, Conv, [128, 3, 1]],
  la(x, md.GhostConv(x[-1], 128, 3, 1))  #  [-1, 1, Conv, [128, 3, 1]],
  la(x, md.GhostConv(x[-1], 128, 3, 1))  #  [-1, 1, Conv, [128, 3, 1]],
  la(x, md.GhostConv(x[-1], 128, 3, 1))  #  [-1, 1, Conv, [128, 3, 1]],
  la(x, md.Concat([x[-1], x[-2], x[-3], x[-4], x[-5], x[-6]], -1))  #  [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
  la(x, md.GhostConv(x[-1], 256, 1, 1))  #  [-1, 1, Conv, [256, 1, 1]], # 63

  la(x, md.GhostConv(x[-1], 128, 1, 1))  #  [-1, 1, Conv, [128, 1, 1]],
  la(x, md.Upsample(x[-1], 2))  #  [-1, 1, nn.Upsample, [None, 2, 'nearest']],
  la(x, md.GhostConv(x[25], 128, 1, 1)) #  [24, 1, Conv, [128, 1, 1]], # route backbone P3
  la(x, md.Concat([x[-1], x[-2]]))  #  [[-1, -2], 1, Concat, [1]],

  la(x, md.GhostConv(x[-1], 128, 1, 1))  #  [-1, 1, Conv, [128, 1, 1]],
  la(x, md.GhostConv(x[-2], 128, 1, 1))  #  [-2, 1, Conv, [128, 1, 1]],
  la(x, md.GhostConv(x[-1], 64, 3, 1))  #  [-1, 1, Conv, [64, 3, 1]],
  la(x, md.GhostConv(x[-1], 64, 3, 1))  #  [-1, 1, Conv, [64, 3, 1]],
  la(x, md.GhostConv(x[-1], 64, 3, 1))  #  [-1, 1, Conv, [64, 3, 1]],
  la(x, md.GhostConv(x[-1], 64, 3, 1))  #  [-1, 1, Conv, [64, 3, 1]],
  la(x, md.Concat([x[-1], x[-2], x[-3], x[-4], x[-5], x[-6]]))  #  [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
  la(x, md.GhostConv(x[-1], 128, 1, 1))  #  [-1, 1, Conv, [128, 1, 1]], # 78

  # la(x, md.MP(x[-1]))  #  [-1, 1, MP, []],
  # la(x, md.GhostConv(x[-1], 128, 1, 1))  #  [-1, 1, Conv, [128, 1, 1]],
  # la(x, md.GhostConv(x[-3], 128, 1, 1))  #  [-3, 1, Conv, [128, 1, 1]],
  # la(x, md.GhostConv(x[-1], 128, 3, 2))  #  [-1, 1, Conv, [128, 3, 2]],
  # la(x, md.Concat([x[-1], x[-3], x[63]]))  #  [[-1, -3, 63], 1, Concat, [1]],

  # la(x, md.GhostConv(x[-1], 256, 1, 1))  #  [-1, 1, Conv, [256, 1, 1]],
  # la(x, md.GhostConv(x[-2], 256, 1, 1))  #  [-2, 1, Conv, [256, 1, 1]],
  # la(x, md.GhostConv(x[-1], 128, 3, 1))  #  [-1, 1, Conv, [128, 3, 1]],
  # la(x, md.GhostConv(x[-1], 128, 3, 1))  #  [-1, 1, Conv, [128, 3, 1]],
  # la(x, md.GhostConv(x[-1], 128, 3, 1))  #  [-1, 1, Conv, [128, 3, 1]],
  # la(x, md.GhostConv(x[-1], 128, 3, 1))  #  [-1, 1, Conv, [128, 3, 1]],
  # la(x, md.Concat([x[-1], x[-2], x[-3], x[-4], x[-5], x[-6]]))  #  [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
  # la(x, md.GhostConv(x[-1], 256, 1, 1))  #  [-1, 1, Conv, [256, 1, 1]], # 88

  # la(x, md.MP(x[-1]))  #  [-1, 1, MP, []],
  # la(x, md.GhostConv(x[-1], 256, 1, 1))  #  [-1, 1, Conv, [256, 1, 1]],
  # la(x, md.GhostConv(x[-3], 256, 1, 1))  #  [-3, 1, Conv, [256, 1, 1]],
  # la(x, md.GhostConv(x[-1], 256, 3, 2))  #  [-1, 1, Conv, [256, 3, 2]],
  # la(x, md.Concat([x[-1], x[-3], x[51]]))  #  [[-1, -3, 51], 1, Concat, [1]],

  # la(x, md.GhostConv(x[-1], 512, 1, 1))  #  [-1, 1, Conv, [512, 1, 1]],
  # la(x, md.GhostConv(x[-2], 512, 1, 1))  #  [-2, 1, Conv, [512, 1, 1]],
  # la(x, md.GhostConv(x[-1], 256, 3, 1))  #  [-1, 1, Conv, [256, 3, 1]],
  # la(x, md.GhostConv(x[-1], 256, 3, 1))  #  [-1, 1, Conv, [256, 3, 1]],
  # la(x, md.GhostConv(x[-1], 256, 3, 1))  #  [-1, 1, Conv, [256, 3, 1]],
  # la(x, md.GhostConv(x[-1], 256, 3, 1))  #  [-1, 1, Conv, [256, 3, 1]],
  # la(x, md.Concat([x[-1], x[-2], x[-3], x[-4], x[-5], x[-6]]))  #  [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
  # la(x, md.GhostConv(x[-1], 512, 1, 1))  #  [-1, 1, Conv, [512, 1, 1]], # 101

  # la(x, md.RepConv(x[75], 256, 3, 1, name='repconv1'))  #  [75, 1, RepConv, [256, 3, 1]], # 102
  # la(x, md.RepConv(x[88], 512, 3, 1, name='repconv2'))  #  [88, 1, RepConv, [512, 3, 1]], # 103
  # la(x, md.RepConv(x[101], 1024, 3, 1, name='repconv3'))  #  [101, 1, RepConv, [1024, 3, 1]], # 104


  # la(x, md.Last([x[102], x[103], x[104]], num_classes))

    #  [[102,103,104], 1, IDetect, [nc, anchors]],   # Detect(P3, P4, P5)
  # regression part
  # la(x, md.RegFC(x[53]))

  la(x, md.RegFlat(x[53]))
  
  # la(x, md.RegFlat(x[75]))

  return x