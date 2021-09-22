#!/usr/bin/env python
import os
import sys
import logging
import numpy as np
import time
import scipy.io
import glob
import pickle

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


all_files=open('whole_data.txt','r')
num_class = 8
scale = 8
for filename in all_files:
    filename=filename.strip()
    tmp=filename.split('/')
    the_id=tmp[-1]
    print(the_id)
    # generate categorical label
    path3='../../../../data/physionet_data/arousal_annotate/'
    path4='../../../../data/physionet_data/new_arousal/'

    set_sleep=set(['W','N1','N2','N3', 'R'])
    # U-0; W-1, N1-2; N2-3; N3-4; R-5; arousal-6; apnea-7

    # use this to determine d1
    arousal_file = path4 + the_id + '-arousal.mat'
    label_ori = import_arousals(arousal_file)
    d1=label_ori.shape[1] # d1 is different for each file, indicates the length of each person's sleep
    d1 = d1//scale
    print(d1)
    #print(set(list(label_ori)))
    
    #TODO: negative for paddings 
    #4096*256
    label_ori = np.zeros((num_class,4096*256))  # pad all to max length
    #label_ori[:, d1:] = -1 #masked
    print(label_ori.shape)

    with open(path3 + the_id + '.txt', 'rb') as fp:
        annotate=pickle.load(fp) # annotate is a list of length 100-1000 or so, different for every id; contain annotations
        #print(annotate)

    location=np.load(path3 + the_id + '.npy') #  np array, location.shape (613,), location.shape[0] = len(annotate)
    location = location//scale #downscale by 8

    anno1=[]  # sleep stages
    anno2=[]  # apnea and arousal
    loca1=[]
    loca2=[]
    for i in range(len(annotate)):
        if annotate[i] in set_sleep:
            anno1.append(annotate[i])
            loca1.append(location[i])
        else:
            anno2.append(annotate[i])
            loca2.append(location[i])

    label_ori[0, :]=1 #undefined as default
    for i in range(len(anno1)):
        start = loca1[i]
        if i == len(anno1)-1:
            end = d1
        else:
            end = loca1[i+1]
        if anno1[i] == 'W':
            cat=1
        elif anno1[i] == 'N1':
            cat=2
        elif anno1[i] == 'N2':
            cat=3
        elif anno1[i] == 'N3':
            cat=4
        elif anno1[i] == 'R':
            cat=5
        else:
            print(anno1[i])
        label_ori[cat,start:end]=1
        label_ori[0,start:end]=0

    # arousal and sleep annotation
    #print(anno2)
    #print(loca2)
    for i in range(0,len(anno2),2):
        # (resp_hypopnea, resp_hypopnea), pairwise
        if 'arousal_rera' in anno2[i]:
            cat = 6
            start=loca2[i]-int(400//scale)
            end=loca2[i+1]+int(2000//scale)
        elif 'arousal' in anno2[i]:
            #print(anno2[i], anno2[i+1])
            cat = 6
            start=loca2[i]-int(400//scale)
            end=loca2[i+1]+int(400//scale)
        else: # apnea, hypopnea
            #print(anno2[i])
            start=loca2[i]
            end=loca2[i+1]
            cat=7
        label_ori[cat,start:end]=1
    
    label_ori[:, d1:] = -1 #masked
    #label_ori = np.hstack(label_ori, -np.ones((num_class,1048576-d1)))
    #label_ori = label_ori[:, :1048576]
    print(label_ori.shape) # (4096*256, 8)
    print(label_ori)
    #print(np.min(np.sum(label_ori, axis=1)))
    # pad 8 numbers at the start in case a class is empty
    #label_ori=np.hstack((np.arange(1,9).reshape(1,-1),label_ori)) # in case a class is empty
    # from 1*n -> one-hot 8*n
    #label = to_categorical(label_ori[0,:], num_classes=None).T
    #label = label[1:,8:].T
    #print('label shape',label.shape)
    #new_label = np.array([np.where(r==1)[0][0] for r in label.T])
    #new_label = np.argmax(label.T, axis=1)+1
    #print(new_label)
    np.save('../../../../data/physionet_data/avg8_8m_multi_task_label/'+the_id +'.npy', label_ori)
