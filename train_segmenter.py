from utils import *
from keras.callbacks import *
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os


x_tr, y_tr, x_va, y_va, x_te, y_te = get_data()
#show_samples(x_tr, y_tr, 3)
model = build_unet(x_tr.shape[1:])
model.compile(optimizer=Adam(lr=0.0001), 
              metrics=['accuracy',iou],
              loss='binary_crossentropy')
mc = ModelCheckpoint('weights/segment.h5', save_best_only=True, save_weights_only=True)
tb = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=16, 
        #write_images=True,
        write_graph=True, 
        write_grads=True)

es = EarlyStopping(patience=9)
datagen = ImageDataGenerator(featurewise_center=True,
    featurewise_std_normalization=True,
    zca_whitening=False,
    #rotation_range=45.,
    #width_shift_range=0.2,
    #height_shift_range=0.2,
    #shear_range=0.05,
    #fill_mode='constant',
    horizontal_flip=True,
    vertical_flip=True,)
datagen.fit(x_tr)
batches = datagen.flow(x_tr,y_tr,16)#,save_to_dir='fg_aug')
batches_val = datagen.flow(x_va,y_va,16)#,save_to_dir='fg_aug')
model.fit_generator(batches, 
          validation_data=batches_val, 
          epochs=50, callbacks=[mc, es, tb])
scores = model.evaluate_generator(batches)
for i in range(len(model.metrics_names)):
    print('Train %s: %f'%(model.metrics_names[i],scores[i]))

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
