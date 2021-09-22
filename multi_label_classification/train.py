from __future__ import print_function
import os
import pickle
import sys
import argparse
from datetime import datetime
import random
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import cv2
from keras.models import Model
from keras.layers import Input, concatenate, Conv1D, MaxPooling1D, Conv2DTranspose,Lambda
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import tensorflow as tf
import keras
import scipy.io
from keras.backend.tensorflow_backend import set_session, clear_session, get_session
import unet as unet

# Reset Keras Session
def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()
    try:
        del classifier # this is from global space - change this as you need
    except:
        pass

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

def generate_data(channel_idx, label_idx, train_line, batch_size, path1, path2, if_train):
    """
    Params
    ------
    channel_idx: list, index of selected channels
    0-6: EEG, F3-M2, F4-M2, C3-M2, C4-M1, O1-M2, O2-M1, E1-M2
    7: Chin
    8: ABD
    9: Chest
    10: Airflow
    11: SaO2
    12: ECG

    label_idx: list, 
    """
    # augmentation parameters
    #if_time=True
    #max_scale=1.15
    #min_scale=1
    if_mag=True
    max_mag=1.15
    min_mag=0.9
    #if_flip=False
    
    i = 0
    while True:
        image_batch = []
        label1_batch = []
        label2_batch = []
        label3_batch = []
        for b in range(batch_size):
            if i == len(train_line):
                i = 0
                random.shuffle(train_line)
            sample = train_line[i]
            i += 1
            the_id=sample.split('/')[-1]
            # channel selections
            image = np.load(path1 + the_id + '.npy')[channel_idx,:]
            
            label = np.load(path2 + the_id + '.npy')[label_idx,:]  #arousal and  TODO: check apnea only
            # make new label
            #idx = np.argwhere(label[0,:] == -1).flatten()
            #label1 = np.argmax(label[0:6,:], axis = 0)
            #print(label1)
            label1 = label[:6,:] #sleep stages
            label2 = label[[6],:] #arousal
            label3 = label[[7],:] #apnea

            #print(the_id, image.shape, label.shape)

            if if_train:
                #rrr=random.random()
                #rrr_scale=rrr*(max_scale-min_scale)+min_scale
                rrr=random.random()
                rrr_mag=rrr*(max_mag-min_mag)+min_mag
                #rrr_flip=random.random()
                #if(if_time):
                #    image=scaleImage(image,rrr_scale)
                #    label=scaleImage(label,rrr_scale)
                if(if_mag):
                    image=image*rrr_mag
                #if(if_flip & (rrr_flip>0.5)):
                #    image=cv2.flip(image,1)
                #    label=cv2.flip(label,1)

            image_batch.append(image.T)
            label1_batch.append(label1.T)
            label2_batch.append(label2.T)
            label3_batch.append(label3.T)

        image_batch=np.array(image_batch)
        label1_batch=np.array(label1_batch) 
        label2_batch=np.array(label2_batch)
        label3_batch=np.array(label3_batch)
        #print(image_batch.shape, label1_batch.shape, label2_batch.shape, label3_batch.shape)
        yield image_batch, {'conv1d_51': label1_batch, 'conv1d_52': label2_batch, 'conv1d_53': label3_batch}  #label1_batch, label2_batch, label3_batch

def train_model(model_path, i, n, channel_idx, label_idx, path1, path2, train_path):
    """
    Params

    """
    # hyperparameters
    size= 4096*256
    #remember to change signal channel selection and weight name
    channel=len(channel_idx)
    labels = ['Undefined', 'Wake', 'N1', 'N2', 'N3', 'REM', 'Arousal', 'Apnea'] 
    labels = [labels[i] for i in label_idx]
    batch_size=4
    num_class = len(labels) #arousal only
    
    #initiate keras
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    set_session(tf.Session(config=config))
    K.set_image_data_format('channels_last')  # TF dimension ordering in this code

    model = unet.get_unet(size, channel, num_class)
    model.summary()
    name_model='weights_' + str(i) + '.h5'

    # read training data files    
    all_ids=open(train_path,'r')
    all_line=[]
    for line in all_ids:
        all_line.append(line.rstrip())
    all_ids.close()

    random.seed(i)
    random.shuffle(all_line)

    partition_ratio=0.8
    train_line=all_line[0:int(len(all_line)*partition_ratio)]
    test_line=all_line[int(len(all_line)*partition_ratio):len(all_line)]

    callbacks = [
            #keras.callbacks.TensorBoard(log_dir='./',
            #histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(os.path.join(model_path, name_model),
            verbose=0, save_weights_only=False,monitor='val_loss')
            ]
    history = model.fit_generator(
            generate_data(channel_idx, label_idx, train_line, batch_size,path1, path2, True),
            steps_per_epoch=int(len(train_line) // batch_size), nb_epoch=n,
            validation_data=generate_data(channel_idx,label_idx, test_line,batch_size, path1, path2,False),
            validation_steps=int(len(test_line) // batch_size),callbacks=callbacks)
    
    pickle.dump(history.history, open('history_'+str(i)+'.pkl', 'wb'))    


def main():
    parser = argparse.ArgumentParser(description = 'sleep model train program',
        usage = 'use "python %(prog)s --help" for more information',
        formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument('-i','--ifold', type=int, 
        help = 'fold number; also used as seed to split train and val')

    parser.add_argument('-n', type=int,
        help = 'number of epoch', default = 40)

    parser.add_argument('-c','--channel_idx', type=int,
        nargs = '+',
        default = [0,1,2,3,4,5,6,7], 
        help = ''' a list, index of selected signals. 0: SaO2, 1: H.R., 2:ECG, 3:THR RES, 4: ABDO RES, 5:POSITION, 6: LIGHT, 7: OXSTAT.
        default: all ''')
    
    parser.add_argument('-l','--label_idx', type=int,
        nargs = '+',
        default = [0,1,2,3,4,5,6,7],
        help = ''' a list, index of selected labels. 0:'Undefined', 1:'Wake', 2:'N1', 3:'N2', 4:'N3', 5:'REM', 6:'Arousal', 7:'Apnea'] 
        default: [0,1,2,3,4,5,6,7] ''')

    parser.add_argument('--sig_path', type=str, default = '../../../data/shhs_data/shhs_image_avg5/',
        help = 'signal path, default = ../../../data/shhs_data/shhs_image_avg5(for shhs1)')

    parser.add_argument('--lab_path', type=str, default = '../../../data/shhs_data/shhs_label_avg5/',
        help = 'label path, default = ../../../data/shhs_data/shhs_label_avg5 (for shhs1)')
    
    parser.add_argument('--train_path', type=str, default = 'whole_train_80.txt',
        help = 'train path, default = whole_train_80.txt')

    args = parser.parse_args()
    opts = vars(args)

    run(**opts)

def run(ifold, n, channel_idx, label_idx, sig_path, lab_path, train_path):
    model_path = './weights/'
    os.makedirs(model_path, exist_ok = True)

    print('Start training fold ', ifold, '... ...')
    print(channel_idx, label_idx)
    train_model(model_path, ifold, n, channel_idx, label_idx, sig_path, lab_path, train_path)
    reset_keras()

if __name__ == '__main__':
    main()
