import glob
import re

import numpy as np
from PIL import Image

imgs = glob.glob('resized_dataset/test/images/*.bmp')
defect_shape = {'A3': [],
                'B3': [],
                'C': [],
                'D': [],
                'E3': []}
angle_defect_shape = {'A3': [],
                      'B3': [],
                      'C': [],
                      'D': [],
                      'E3': []}

magnet_current = {0.2: [],
                  0.25: [],
                  0.3: [],
                  0.35: [],
                  0.4: [],
                  0.45: []}
angle_magnet_current = {0.2: [],
                        0.25: [],
                        0.3: [],
                        0.35: [],
                        0.4: [],
                        0.45: []}

def img_to_np(filename):
    im = Image.open(filename)
    arr = np.asarray(im)
    return arr

def angle_correction(angle):
    angle = 90 - abs(angle%180 - 90)
    return angle

for i, img in enumerate(imgs):
    arr = img_to_np(img)
    data = re.search("images\/([A-Z1-9]*)_(\d.\d*)_(\d*)", img)
    defect, current, angle = data.group(1), float(data.group(2)), int(data.group(3))
    angle = angle_correction(angle)
    if defect in list(defect_shape.keys()):
        defect_shape[defect].append(arr)
        angle_defect_shape[defect].append(angle)
    if current in list(magnet_current.keys()):
        magnet_current[current].append(arr)
        angle_magnet_current[current].append(angle)
    print(f"{i+1}/{len(imgs)}")

for keys in list(defect_shape.keys()):
    np_arr = np.array(defect_shape[keys])
    np.save(f"np_data/{keys}_arr.npy", np_arr)
    np_ang = np.array(angle_defect_shape[keys])
    np.save(f"np_data/{keys}_angle.npy", np_ang)

for keys in list(magnet_current.keys()):
    np_arr = np.array(magnet_current[keys])
    np.save(f"np_data/current_{keys}_arr.npy", np_arr)
    np_ang = np.array(angle_magnet_current[keys])
    np.save(f"np_data/current_{keys}_angle.npy", np_ang)