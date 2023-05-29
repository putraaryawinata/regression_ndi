from PIL import Image
import numpy as np
import glob
import re

def im_to_nparr(image_name):
    im = Image.open(image_name)
    arr = np.asarray(im)
    return arr

def xy_alldata(image_list):
    x = []
    y = []
    total = len(image_list)
    for i, image in enumerate(image_list):
        arr = im_to_nparr(image)
        data = re.search("_(0\.[\d]*)_([\d]*)", image)
        # intensity = float(data.group(1))
        # arr = np.append(hist_arr, intensity)
        angle = float(data.group(2)) % 180
        l_angle = 90 - abs(angle - 90)
        x.append(arr)
        y.append(l_angle)
        if i%10000 == 0:
            print(f"{i}/{total}")
    return x, y

def load(path="resized_dataset/train/images/*.bmp"):
    x, y = xy_alldata(glob.glob(path))
    return np.array(x), np.array(y)

def expand(obj):
    if isinstance(obj, list):
        for i, x in enumerate(obj):
            obj[i] = np.expand_dims(x, axis=-1)
        return obj
    if isinstance(obj, np.ndarray):
        return np.expand_dims(obj, axis=-1)
    raise("Please use numpy array format or list of it!")

def save(obj, dir):
    try:
        if isinstance(obj, list):
            for i, x in enumerate(obj):
                obj[i] = np.save(f"{dir}/{i}.npy", x)
            print("saved")
        if isinstance(obj, np.ndarray):
            np.save(f"{dir}/numpy.npy", obj)
            print(f"saved")
    except:
        raise("Please use numpy array format or list of it!")