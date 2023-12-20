from __future__ import division
import cv2
import numpy as np

def preprocess_imgs(x):
    x = x.astype(np.float32)
    x[..., 0] -= 103.939
    x[..., 1] -= 116.779
    x[..., 2] -= 123.68
    return x

def preprocess_label(paths):
    maps=np.zeros((len(paths), 1))
    for i, path in enumerate(paths):
        label = float(path)
        maps[i] = label
    return maps


def preprocess_images(srimgpaths,hrimgpaths,saliencypaths,shape_r,shape_c):
    srimgs =[]
    hrimgs = []
    salimgs = []

    for patha,pathb,pathc in zip(srimgpaths,hrimgpaths,saliencypaths):
        srimage = cv2.imread(patha,1)
        srimage = preprocess_imgs(srimage)
        srimage = cv2.resize(srimage, (shape_c, shape_r))
        srimage = np.asarray(srimage,np.float32)

        hrimage = cv2.imread(pathb,1)
        hrimage = preprocess_imgs(hrimage)
        hrimage = cv2.resize(hrimage, (shape_c, shape_r))
        hrimage = np.asarray(hrimage,np.float32)

        salimage = cv2.imread(pathc, 0)
        salimage = np.asarray(salimage, np.float32)
        salimage /= 255.0
        salimage = cv2.resize(salimage, (16, 12)) # change based on your image sizes
        salimage = np.expand_dims(salimage, axis=-1)

        srimgs.append(srimage)
        hrimgs.append(hrimage)
        salimgs.append(salimage)

    srimgs=np.array(srimgs)
    hrimgs=np.array(hrimgs)
    salimgs=np.array(salimgs)

    return srimgs,hrimgs,salimgs

