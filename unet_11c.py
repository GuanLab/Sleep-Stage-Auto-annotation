from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv1D, MaxPooling1D, Conv2DTranspose,Lambda
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import keras
import tensorflow as tf
from keras.layers import ZeroPadding1D

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

num_class=8
size= 4096*128*2 #sometimes changed.. be careful this match the size in train or predict file in use
channel=11
batch_size=32
ss=10


def crossentropy_cut(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f= tf.clip_by_value(y_pred_f, 1e-7, (1. - 1e-7))
    mask=K.cast(K.greater_equal(y_true_f,-0.5),dtype='float32')
    out = -(y_true_f * K.log(y_pred_f)*mask + (1.0 - y_true_f) * K.log(1.0 - y_pred_f)*mask)
    out=K.mean(out)
    return out

#def categorical_crossentropy_cut(y_true,y_pred):
#    y_true_f = K.flatten(y_true)
#    y_pred_f = K.flatten(y_pred)
#    y_pred_f= tf.clip_by_value(y_pred_f, 1e-7, (1. - 1e-7))
#    mask = K.cast(K.greater_equal(y_true_f,-0.5),dtype='float32')
#    out = -(y_true_f * K.log(y_pred_f) * mask)
#    out=K.mean(out)
#    return out

def weighted_categorical_crossentropy_cut(y_true,y_pred):
    l0=crossentropy_cut(y_true[:,:,0],y_pred[:,:,0])
    l1=crossentropy_cut(y_true[:,:,1],y_pred[:,:,1])
    l2=crossentropy_cut(y_true[:,:,2],y_pred[:,:,2])
    l3=crossentropy_cut(y_true[:,:,3],y_pred[:,:,3])
    l4=crossentropy_cut(y_true[:,:,4],y_pred[:,:,4])
    l5=crossentropy_cut(y_true[:,:,5],y_pred[:,:,5])
    l6=crossentropy_cut(y_true[:,:,6],y_pred[:,:,6])
    l7=crossentropy_cut(y_true[:,:,7],y_pred[:,:,7])
    out = (l0*7 + l1 + l2 + l3 + l4 + l5 + l6 + l7)/(7+7)
    return out

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)

def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same'):
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    mask=K.cast(K.greater_equal(y_true_f,-0.5),dtype='float32')
    intersection = K.sum(y_true_f * y_pred_f * mask)
    return (2. * intersection + ss) / (K.sum(y_true_f * mask) + K.sum(y_pred_f * mask) + ss)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((size, channel)) #4096*128
    print(inputs.shape)

    conv01 = Conv1D(32, 7, activation='relu', padding='same')(inputs)
    conv01 = Conv1D(32, 7, activation='relu', padding='same')(conv01)
    pool01 = MaxPooling1D(pool_size=2)(conv01) #4096*64

    conv0 = Conv1D(40, 7, activation='relu', padding='same')(pool01)#+8
    conv0 = Conv1D(40, 7, activation='relu', padding='same')(conv0)
    pool0 = MaxPooling1D(pool_size=2)(conv0) #4096*32

    conv1 = Conv1D(48, 7, activation='relu', padding='same')(pool0)#+8
    conv1 = Conv1D(48, 7, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv1) #4096*16

    conv2 = Conv1D(64, 7, activation='relu', padding='same')(pool1)#+16
    conv2 = Conv1D(64, 7, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling1D(pool_size=2)(conv2) #4096*8

    conv3 = Conv1D(80, 7, activation='relu', padding='same')(pool2)#+16
    conv3 = Conv1D(80, 7, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling1D(pool_size=2)(conv3) #4096*4

    conv4 = Conv1D(112, 7, activation='relu', padding='same')(pool3)#+32
    conv4 = Conv1D(112, 7, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling1D(pool_size=2)(conv4) #4096*2

    conv5 = Conv1D(144, 7, activation='relu', padding='same')(pool4)#+32
    conv5 = Conv1D(144, 7, activation='relu', padding='same')(conv5)
    pool5 = MaxPooling1D(pool_size=2)(conv5) #4096

    conv6 = Conv1D(208, 7, activation='relu', padding='same')(pool5)#+64
    conv6 = Conv1D(208, 7, activation='relu', padding='same')(conv6)
    pool6 = MaxPooling1D(pool_size=2)(conv6) #2048

    conv7 = Conv1D(272, 7, activation='relu', padding='same')(pool6)#+64 
    conv7 = Conv1D(272, 7, activation='relu', padding='same')(conv7)
    pool7 = MaxPooling1D(pool_size=2)(conv7) #1024

    conv8 = Conv1D(400, 7, activation='relu', padding='same')(pool7)#+128
    conv8 = Conv1D(400, 7, activation='relu', padding='same')(conv8)
    pool8 = MaxPooling1D(pool_size=2)(conv8) #512

    conv9 = Conv1D(528, 7, activation='relu', padding='same')(pool8)#+128
    conv9 = Conv1D(528, 7, activation='relu', padding='same')(conv9)
    pool9 = MaxPooling1D(pool_size=2)(conv9) #256

    conv10 = Conv1D(1024, 7, activation='relu', padding='same')(pool9)#+496
    conv10 = Conv1D(1024, 7, activation='relu', padding='same')(conv10)

    up11 = concatenate([Conv1DTranspose(conv10,528, 2, strides=2, padding='same'), conv9], axis=2)
    conv11 = Conv1D(528, 7, activation='relu', padding='same')(up11)
    conv11 = Conv1D(528, 7, activation='relu', padding='same')(conv11) #512

    up12 = concatenate([Conv1DTranspose(conv11,400, 2, strides=2, padding='same'), conv8], axis=2)
    conv12 = Conv1D(400, 7, activation='relu', padding='same')(up12)
    conv12 = Conv1D(400, 7, activation='relu', padding='same')(conv12) #1024

    up13 = concatenate([Conv1DTranspose(conv12,272, 2, strides=2, padding='same'), conv7], axis=2)
    conv13 = Conv1D(272, 7, activation='relu', padding='same')(up13)
    conv13 = Conv1D(272, 7, activation='relu', padding='same')(conv13) #2048

    up14 = concatenate([Conv1DTranspose(conv13,208, 2, strides=2, padding='same'), conv6], axis=2)
    conv14 = Conv1D(208, 7, activation='relu', padding='same')(up14)
    conv14 = Conv1D(208, 7, activation='relu', padding='same')(conv14) #4096

    up15 = concatenate([Conv1DTranspose(conv14,144, 2, strides=2, padding='same'), conv5], axis=2)
    conv15 = Conv1D(144, 7, activation='relu', padding='same')(up15)
    conv15 = Conv1D(144, 7, activation='relu', padding='same')(conv15) #4096*2

    up16 = concatenate([Conv1DTranspose(conv15,112, 2, strides=2, padding='same'), conv4], axis=2)
    conv16 = Conv1D(112, 7, activation='relu', padding='same')(up16)
    conv16 = Conv1D(112, 7, activation='relu', padding='same')(conv16) #4096*4

    up17 = concatenate([Conv1DTranspose(conv16,80, 2, strides=2, padding='same'), conv3], axis=2)
    conv17 = Conv1D(80, 7, activation='relu', padding='same')(up17)
    conv17 = Conv1D(80, 7, activation='relu', padding='same')(conv17) #4096*8

    up18 = concatenate([Conv1DTranspose(conv17,64, 2, strides=2, padding='same'), conv2], axis=2)
    conv18 = Conv1D(64, 7, activation='relu', padding='same')(up18)
    conv18 = Conv1D(64, 7, activation='relu', padding='same')(conv18) #4096*16
    
    up19 = concatenate([Conv1DTranspose(conv18,48, 2, strides=2, padding='same'), conv1], axis=2)
    conv19 = Conv1D(48, 7, activation='relu', padding='same')(up19)
    conv19 = Conv1D(48, 7, activation='relu', padding='same')(conv19) #4096*32

    up20 = concatenate([Conv1DTranspose(conv19,40, 2, strides=2, padding='same'), conv0], axis=2)
    conv20 = Conv1D(40, 7, activation='relu', padding='same')(up20)
    conv20 = Conv1D(40, 7, activation='relu', padding='same')(conv20) #4096*64

    up21 = concatenate([Conv1DTranspose(conv20,32, 2, strides=2, padding='same'), conv01], axis=2)
    conv21 = Conv1D(32, 7, activation='relu', padding='same')(up21)
    conv21 = Conv1D(32, 7, activation='relu', padding='same')(conv21) #4096*128

    conv22 = Conv1D(num_class, 1, activation='sigmoid')(conv21) # batch, size, channel

    model = Model(inputs=[inputs], outputs=[conv22])

    model.compile(optimizer=Adam(lr=1e-4,beta_1=0.9, beta_2=0.999,decay=1e-5), \
        loss=weighted_categorical_crossentropy_cut, metrics=[dice_coef])

    return model


