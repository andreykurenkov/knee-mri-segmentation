#!/usr/bin/python
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.losses import *
from keras.preprocessing.image import *
from os.path import isfile
from tqdm import tqdm
import random
import skimage.io as io
import skimage.morphology as mo
import matplotlib.pyplot as plt
from PIL import Image
from resizeimage import resizeimage

# img helper functions

def print_info(x):
    print(str(x.shape) + ' - Min: ' + str(x.min()) + ' - Mean: ' + str(x.mean()) + ' - Max: ' + str(x.max()))

def show_samples(x, y, num):
    rnd = np.random.permutation(len(x))
    for i in range(num):
        plt.figure(figsize=(15, 5))
        plt.subplot(1,2,1)
        img = x[rnd[i]] 
        plt.axis('off')
        plt.imshow(img)
        plt.subplot(1,2,2)
        img = y[rnd[i]]
        plt.axis('off')
        plt.imshow(np.repeat(img,3,2))
        plt.show()

def shuffle(x, y):
    perm = np.random.permutation(len(x))
    x = x[perm]
    y = y[perm]
    return x, y

def split(x, y, tr_size=0.8, va_size=0.1):
    tr_size = int(len(x) * tr_size)
    va_size = int(len(x) * tr_size)
    va_start = tr_size + va_size
    x_tr = x[:tr_size]
    y_tr = y[:tr_size]
    x_va = x[tr_size:va_start]
    y_va = y[tr_size:va_start]
    x_te = x[va_start:]
    y_te = y[va_start:]
    return x_tr, y_tr, x_va, y_va, x_te, y_te

def read_image(path, size=None, norm=False, grey=False):
    fd_img = open(path, 'rb')
    img = Image.open(fd_img)
    if size is not None:
        resized = resizeimage.resize_contain(img, size)
        if not grey:
            img = np.array(resized.convert("RGB"))
        else:
            img = np.array(resized.convert("L"))[...,np.newaxis]
    fd_img.close()
    img = (img - img.mean()) / img.std() if norm else img
    return img

def load_data(path, size=(24,224,224), norm=False):
    train_files = [x for x in os.listdir(path) if 'img' in x]
    x, y = [], []
    for i in tqdm(range(len(train_files))):
        x_path = os.path.join(path,train_files[i])
        img = read_image(x_path, size=size, norm=norm)
        x.append(img)
        y_path = x_path.replace('img','mask')
        img = read_image(y_path, size=size, norm=norm, grey=True)>0.5
        y.append(img)
    x = np.array(x)
    y = np.array(y)
    return x, y

def print_weights(weight_file_path):
    """
    Prints out the structure of HDF5 file.

    Args:
      weight_file_path (str) : Path to the file to analyze
    """
    f = h5py.File(weight_file_path)
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))

        if len(f.items())==0:
            return 

        for layer, g in f.items():
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))

            print("    Dataset:")
            for p_name in g.keys():
                param = g[p_name]
                print("      {}: {}".format(p_name, param.shape)) #try only "param"
    finally:
        f.close()

# Models

def conv_block(m, dim, acti, bn, res, do=0):
    n = Conv2D(dim, 3, activation=acti, padding='same')(m)
    n = BatchNormalization()(n) if bn else n
    n = Dropout(do)(n) if do else n
    n = Conv2D(dim, 3, activation=acti, padding='same')(n)
    n = BatchNormalization()(n) if bn else n
    return Add()([m, n]) if res else n

def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
    if depth > 0:
        n = conv_block(m, dim, acti, bn, res)
        m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
        m = level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, mp, up, res)
        if up:
            m = UpSampling2D()(m)
            m = Conv2D(dim, 2, activation=acti, padding='same')(m)
        else:
            m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
        n = Add()([n, m])
        m = conv_block(n, dim, acti, bn, res)
    else:
        m = conv_block(m, dim, acti, bn, res, do)
    return m

def UNet(img_shape, out_ch=1, start_ch=32, depth=4, inc_rate=1., activation='elu', 
         dropout=0.5, batchnorm=False, maxpool=False, upconv=True, residual=False):
    i = Input(shape=img_shape)
    o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
    o = Conv2D(out_ch, 1, activation='sigmoid')(o)
    return Model(inputs=i, outputs=o)

# Loss Functions

# 2TP / (2TP + FP + FN)
def f1(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)

def f1_np(y_true, y_pred):
    return (2. * (y_true * y_pred).sum() + 1.) / (y_true.sum() + y_pred.sum() + 1.)

def f1_loss(y_true, y_pred):
    return 1-f1(y_true, y_pred)

def f2(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (5. * intersection + 1.) / (4. * K.sum(y_true_f) + K.sum(y_pred_f) + 1.)

def f2_loss(y_true, y_pred):
    return 1-f2(y_true, y_pred)

dice = f1
dice_loss = f1_loss

def iou(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1. - intersection)

def iou_np(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1.) / (y_true.sum() + y_pred.sum() + 1. - intersection)

def iou_loss(y_true, y_pred):
    return -iou(y_true, y_pred)

def precision(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.) / (K.sum(y_pred_f) + 1.)

def precision_np(y_true, y_pred):
    return ((y_true * y_pred).sum() + 1.) / (y_pred.sum() + 1.)

def recall(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.) / (K.sum(y_true_f) + 1.)

def recall_np(y_true, y_pred):
    return ((y_true * y_pred).sum() + 1.) / (y_true.sum() + 1.)

def mae_img(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return mae(y_true_f, y_pred_f)

def bce_img(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return binary_crossentropy(y_true_f, y_pred_f)

def f1_bce(y_true, y_pred):
    return f1_loss(y_true, y_pred) + bce_img(y_true, y_pred)

# FP + FN
def error(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return K.sum(K.abs(y_true_f - y_pred_f)) / float(224*224)

def error_np(y_true, y_pred):
    return (abs(y_true - y_pred)).sum() / float(len(y_true.flatten()))
