from utils import *
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os


if os.path.isfile('data.npz'):
    data = np.load('data.npz')
    x_tr = data['x_tr']
    y_tr = data['y_tr']
    x_va = data['x_va']
    y_va = data['y_va']
else:
    x, y = load_data('fg', size=(224,224), norm=False)
    x_tr, y_tr, x_va, y_va, x_te, y_te = split(x,y)
    #np.savez('data.npz', x_tr=x_tr, y_tr=y_tr, x_va=x_va, y_va=y_va, x_te=x_te, y_te=y_te)

#show_samples(x_tr, y_tr, 3)
#x_tr, y_tr = augment(x_tr, y_tr, h_shift=[24,-24], h_flip=False, v_flip=False)
model = UNet(x_tr.shape[1:], 1, 32, 4, 1, 'elu', upconv=False)
model.compile(optimizer=Adam(lr=0.001), loss=f1_loss)
mc = ModelCheckpoint('weights/segment.h5', save_best_only=True, save_weights_only=True)
es = EarlyStopping(patience=9)
datagen = ImageDataGenerator(featurewise_center=True,
    samplewise_center=False,
    featurewise_std_normalization=True,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=45.,
    width_shift_range=20.,
    height_shift_range=20.,
    shear_range=0.05,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=True,
    vertical_flip=True,)
datagen.fit(x_tr)
batches = datagen.flow(x_tr,y_tr,8)#,save_to_dir='fg_aug')
model.fit_generator(batches, 
          validation_data=(x_va, y_va), 
          epochs=10, callbacks=[mc, es])
