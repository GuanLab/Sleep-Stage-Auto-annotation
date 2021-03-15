#!/usr/bin/env python
import os
import sys
import logging
import numpy as np
import cv2
import time
import scipy.io
import glob
import pickle
from keras.utils.np_utils import to_categorical
from keras import backend as K
import tensorflow as tf
import keras
# import cv2
from keras.backend.tensorflow_backend import set_session

# set fraction of GPU the program can use
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

# set the value of data format convention
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

# read into arousal data (arousal data is only helper data, help to determine the sleep length)

def import_arousals(file_name): # target
    import h5py
    import numpy as np
    f = h5py.File(file_name, 'r')
    arousals = np.transpose(np.array(f['data']['arousals']))
    return arousals

# load matlab file, return a dictionary
def import_signals(file_name): # feature
    return scipy.io.loadmat(file_name)['val']
#
def label_major_vote(input_data,scale_pool):
    size_new=int(input_data.shape[1]/scale_pool)
    input_data=input_data.reshape(size_new,scale_pool).T
    input_data=input_data.astype(int)  # bincount need non-negative, int dtype
    counts=np.apply_along_axis(lambda x: np.bincount(x, minlength=9), axis=0, arr=input_data)
    major=np.apply_along_axis(lambda x: np.argmax(x), axis=0, arr=counts)
    major=major.reshape(1,len(major))
    return major

# arrays of pred_pos and pred_neg hist values + input data examination
def score_record(truth, predictions, input_digits=None):
    if input_digits is None: # bin resolution
        input_digits = 3
    scale=10**input_digits
    pos_values = np.zeros(scale + 1, dtype=np.int64)
    neg_values = np.zeros(scale + 1, dtype=np.int64)
    b = scale+1
    r = (-0.5 / scale, 1.0 + 0.5 / scale)
    all_values = np.histogram(predictions, bins=b, range=r)[0] # an array containing values of histogram
    if np.sum(all_values) != len(predictions):
        raise ValueError("invalid values in 'predictions'")
    pred_pos = predictions[truth > 0] # positive/negative true predictions
    pos_values = np.histogram(pred_pos, bins=b, range=r)[0] # array of pred_pos hist value
    pred_neg = predictions[truth == 0] # positive/negative false predictions
    neg_values = np.histogram(pred_neg, bins=b, range=r)[0] # array of pred_neg hist value
    return (pos_values, neg_values)

# calculate auroc & auprc
def calculate_auc(pos_values,neg_values): # auc & auprc; adapted from score2018.py
    # initialize variables
    tp = np.sum(pos_values)
    fp = np.sum(neg_values)
    tn = fn = 0
    tpr = 1
    tnr = 0
    if tp == 0 or fp == 0:
        # If either class is empty, scores are undefined.
        return (float('nan'), float('nan'))
    ppv = float(tp) / (tp + fp)
    auroc = 0
    auprc = 0
    # move threshold by bins and calculate integral
    for (n_pos, n_neg) in zip(pos_values, neg_values):
        tp -= n_pos
        fn += n_pos
        fp -= n_neg
        tn += n_neg
        tpr_prev = tpr
        tnr_prev = tnr
        ppv_prev = ppv
        tpr = float(tp) / (tp + fn)
        tnr = float(tn) / (tn + fp)
        if tp + fp > 0:
            ppv = float(tp) / (tp + fp)
        else:
            ppv = ppv_prev
        auroc += (tpr_prev - tpr) * (tnr + tnr_prev) * 0.5
        auprc += (tpr_prev - tpr) * ppv_prev
    return (auroc, auprc)


###### PARAMETER ###############

#channel=8
num_class=8
size=4096*128*2
num_channel=11
import unet_11c as unet
auc_auprc=open('auc_auprc_physionet_11c_train80_40epoch.txt','w')
model_path = '/local/disk2/xueqing/2019/physionet/code_exp/multi_task/physionet_8m/11c/weights_8m_11c_train80_40epoch/'
#remember to change signals, weight names, auroc file names for different channels 
write_vec=True # whether generate .vec prediction file
reso_digits=3 # auc resolution
batch=1
num_pool = 3
################################
# initialize
scale_pool=2**num_pool
scale=10**reso_digits
positives_all = np.zeros((8, scale + 1), dtype=np.int64)
negatives_all = np.zeros((8, scale + 1), dtype=np.int64)

# load models
if __name__ == '__main__':
    model0 = unet.get_unet()
    model0.load_weights(model_path + 'weights_11c_train80_40epoch_1.h5')

    model1 = unet.get_unet()
    model1.load_weights(model_path + 'weights_11c_train80_40epoch_2.h5')

    model2 = unet.get_unet()
    model2.load_weights(model_path + 'weights_11c_train80_40epoch_3.h5')

    model3 = unet.get_unet()
    model3.load_weights(model_path + 'weights_11c_train80_40epoch_4.h5')

    model4 = unet.get_unet()
    model4.load_weights(model_path + 'weights_11c_train80_40epoch_5.h5')
    #a=np.zeros((1,4096,13))
# load files
    path1='/local/disk2/xueqing/2019/physionet/data/physionet_data/avg8_8m_anchor555/'
    path2='/local/disk2/xueqing/2019/physionet/data/physionet_data/avg8_8m_multi_task_label/'
    all_test_files=open('whole_test_20.txt','r')
    for filename in all_test_files:
        filename=filename.strip()
        tmp=filename.split('/')
        the_id=tmp[-1]
        print(the_id)

        # select signal channels
        signal = np.load(path1 + the_id + '.npy')[0:11,:]
        # signal.shape (11,1048576)
        label = np.load(path2 + the_id + '.npy') # old version
        # label.shape (8,1048576)

        input_pred=np.reshape(signal.T,(batch,size,num_channel)) # error, batch = 2 would work, how to determine batch and size?

        # use model to predict, 5 predicts/model then use average as output
        output = np.zeros((size*batch, num_class))
        output_ori = model0.predict(input_pred)
        output = output + np.reshape(output_ori,(size*batch,num_class))
        output_ori = model1.predict(input_pred)
        output = output + np.reshape(output_ori,(size*batch,num_class))
        output_ori = model2.predict(input_pred)
        output = output + np.reshape(output_ori,(size*batch,num_class))
        output_ori = model3.predict(input_pred)
        output = output + np.reshape(output_ori,(size*batch,num_class))
        output_ori = model4.predict(input_pred)
        output = output + np.reshape(output_ori,(size*batch,num_class))
        output_new = (1/5) * output
        # output_new.shape (1048576,8), contain predics for all 8 classes

        print('label: ', label.shape)
        print('output: ', output_new.shape)
        # calculate auc/auprc
        for i in range(0,8):
            # positives, negatives = score_record(label.T[:,i].flatten(),output_new[:,i].flatten(),reso_digits)
            positives, negatives = score_record(label.T[:,i],output_new[:,i],reso_digits)
            positives_all[i, :] += positives
            negatives_all[i, :] += negatives # (8, scale+1), each row stores the data for one class
            auc, auprc = calculate_auc(positives, negatives)
            auc_auprc.write('%s' % the_id)
            auc_auprc.write('\t%d' % i)
            auc_auprc.write('\t%.3f' % auc)
            auc_auprc.write('\t%.3f' % auprc)
        auc_auprc.write('\n')
        auc_auprc.flush()
        print(auc,auprc)

        # select the part with label
        output_all=output_new

        # write the output to file 1~8
        if(write_vec):
            spec = open(the_id + '_spec' + '.vec','w')
            output_max = output_all.max(axis = 1)
            for i in range(output_all.shape[0]):
                place = np.where(output_all[i,:] == output_max[i])
                place_num = place[0][0]
                spec.write("%d\n" % (place_num+1))
            spec.close()

    all_test_files.close()
    # calculate overall auc & auprc for each class
    for i in range(0, 8):
        auc, auprc = calculate_auc(positives_all[i,:], negatives_all[i, :])
        auc_auprc.write('%s' % 'overall')
        auc_auprc.write('\t%d' % i)
        auc_auprc.write('\t%.3f' % auc)
        auc_auprc.write('\t%.3f' % auprc)
        auc_auprc.write('\n')
    auc_auprc.close()
