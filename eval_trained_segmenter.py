
from utils import *
from keras.callbacks import *
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

x_tr, y_tr, x_va, y_va, x_te, y_te = get_data()
model = UNet(x_tr.shape[1:], 1, 32, 2, 1, 'elu', upconv=False)
model.load_weights("weights/segment.h5")

datagen = ImageDataGenerator(featurewise_center=True,
    featurewise_std_normalization=True)
datagen.fit(x_tr)

show_num = 3
idx = np.random.randint(0,x_tr.shape[0],show_num)
train_inps = x_tr[idx]
train_outs = y_tr[idx]
batches = datagen.flow(train_inps,train_inps,shuffle=False)
aug_inp = next(batches)
out = model.predict(aug_inp[0])
for i in range(show_num):
    show_triple(train_inps[i],out[i],train_outs[i])

#batches = datagen.flow(x_te,y_te,8)#,save_to_dir='fg_aug')
#model.evaluate_generator(batches)
#test_inps = np.random.choose(x_tr,3)
