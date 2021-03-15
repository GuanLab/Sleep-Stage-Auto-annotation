from __future__ import print_function

import os
import sys
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv1D, MaxPooling1D, Conv2DTranspose,Lambda
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import tensorflow as tf
import keras
#import cv2
import scipy.io
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9 
set_session(tf.Session(config=config))

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

size= 4096*256
num_pool=3
channel=11
import unet_11c as unet
train_file = 'whole_train_80.txt' #'whole_train.dat'
train_val_partition_ratio=0.8
model_name ='weights_11c_train80_40epoch_' + sys.argv[1] + '.h5'
batch_size=4
scale_pool=2**num_pool

def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same'):
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x

def crossentropy_cut(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    mask=K.cast(K.greater_equal(y_true_f,-0.5),dtype='float32')
    out = -(y_true_f * K.log(y_pred_f)*mask + (1.0 - y_true_f) * K.log(1.0 - y_pred_f)*mask)
    out=K.mean(out)
    return out

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    ss=1
    mask=K.cast(K.greater_equal(y_true_f,-0.5),dtype='float32')
    intersection = K.sum(y_true_f * y_pred_f*mask )
    return (2. * intersection + ss) / (K.sum(y_true_f*mask ) + K.sum(y_pred_f*mask) + ss)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def scaleImage (image,scale):
    [x,y]= image.shape
    x1=x
    y1=int(round(y*scale))
    image=cv2.resize(image.astype('float32'),(y1,x1)) # check this for multiple channnels!!
    new=np.zeros((x,y))
    if (y1>y):
        start=int(round(y1/2-y/2))
        end=start+y
        new=image[:,start:end]
    else:
        new_start=int(round(y-y1)/2)
        new_end=new_start+y1
        new[:,new_start:new_end]=image
    return new

model = unet.get_unet()
#model.load_weights('weights_' + sys.argv[1] + '.h5')
#model.summary()

#import GS_split
#(train_line,test_line)=GS_split.GS_split('train_gs.dat',size,'whole_train.dat')

from datetime import datetime
import random
path1='/local/disk2/xueqing/2019/physionet/data/physionet_data/avg8_8m_anchor555/'
path2='/local/disk2/xueqing/2019/physionet/data/physionet_data/avg8_8m_multi_task_label/'
all_ids=open(train_file,'r')
all_line=[]
for line in all_ids:
    all_line.append(line.rstrip())
all_ids.close()

random.seed(int(sys.argv[1]))
random.shuffle(all_line)
partition_ratio=train_val_partition_ratio
train_line=all_line[0:int(len(all_line)*partition_ratio)]
test_line=all_line[int(len(all_line)*partition_ratio):len(all_line)]
random.seed(datetime.now())

def generate_data(train_line, batch_size, if_train):
    """Replaces Keras' native ImageDataGenerator."""
##### augmentation parameters ######
    if_time=False
    max_scale=1.15
    min_scale=1
    if_mag=True
    max_mag=1.15
    min_mag=0.9
    if_flip=False
####################################
    i = 0
    while True:
        image_batch = []
        label_batch = []
        for b in range(batch_size):
            if i == len(train_line):
                i = 0
                random.shuffle(train_line)
            sample = train_line[i]
            i += 1
            the_id=sample.split('/')[-1]
            # change this line for different channel
            image = np.load(path1 + the_id + '.npy')[0:11,:]
            label = np.load(path2 + the_id + '.npy')

#            rrr=random.random()
#            if (rrr>0.5):
#                image = np.load(path3 + the_id + '.npy')[0:11,:]
#                label = np.load(path4 + the_id + '.npy')
#            else:
#                image = np.load(path1 + the_id + '.npy')[0:11,:]
#                label = np.load(path2 + the_id + '.npy')

            if (if_train==1):
                rrr=random.random()
                rrr_scale=rrr*(max_scale-min_scale)+min_scale
                rrr=random.random()
                rrr_mag=rrr*(max_mag-min_mag)+min_mag
                rrr_flip=random.random()
                if(if_time):
                    image=scaleImage(image,rrr_scale)
                    label=scaleImage(label,rrr_scale)
                if(if_mag):
                    image=image*rrr_mag
                if(if_flip & (rrr_flip>0.5)):
                    image=cv2.flip(image,1)
                    label=cv2.flip(label,1)

            image_batch.append(image.T)
            label_batch.append(label.T)

        image_batch=np.array(image_batch)
        label_batch=np.array(label_batch)
#        print(image_batch.shape,label_batch.shape)
        yield image_batch, label_batch

#model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=False)
name_model=model_name
callbacks = [
#    keras.callbacks.TensorBoard(log_dir='./',
#    histogram_freq=0, write_graph=True, write_images=False),
    keras.callbacks.ModelCheckpoint(os.path.join('./', name_model),
    verbose=0, save_weights_only=False,monitor='val_loss')
    ]

model.fit_generator(
    generate_data(train_line, batch_size,True),
    steps_per_epoch=int(len(train_line) // batch_size), nb_epoch=40,
    validation_data=generate_data(test_line,batch_size,False),
    validation_steps=int(len(test_line) // batch_size),callbacks=callbacks)

