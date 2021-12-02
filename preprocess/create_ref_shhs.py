import sys 
import numpy as np
import scipy.io
import glob
import os
import cv2
import mne

path1='./'
path2='/nfs/disk3/hyangl/2018/physionet/data/shhs/'

id_all=[]
f=open('./ids_shhs1_all.txt','r')
for line in f:
    id_all.append(line.rstrip())
f.close()

id_all_1000 = []

# note: 13 channels in total
size=5550000
ref=np.zeros((8,size)).astype('float32') # 8 channels for all
num = 0
ref_eeg=np.zeros((2,size)).astype('float32') # 2 channels for EEG

# sort -> resize
i = 0
while num <= 1000:
    the_id = id_all[i]
    the_id=the_id.rstrip()
    i+=1
    print(the_id,num)
    edf = mne.io.read_raw_edf(path2 + the_id + '.edf', verbose=False)
    name_chan = edf.info['ch_names']

    # all 8 channels
    try:
        index_sao2 = name_chan.index('SaO2')
        index_hr = name_chan.index('H.R.')
        index_ecg = name_chan.index('ECG')
        index_thor = name_chan.index('THOR RES')
        index_abd = name_chan.index('ABDO RES')
        index_posi = name_chan.index('POSITION')
        index_light = name_chan.index('LIGHT')
        index_ox = name_chan.index('OX stat')
        # 2 eeg channels
        index_eeg = name_chan.index('EEG')
        index_eeg2 = name_chan.index('EEG(sec)')
    except:
        print('channel not found')
        continue
    else:
        
        image = edf.get_data()[[index_sao2, index_hr, index_ecg, index_thor, index_abd, index_posi, index_light, index_ox],:]
        d0=image.shape[0] #should be d0 = 8
        image.sort(axis=1) 
        image=cv2.resize(image,(size,d0),interpolation=cv2.INTER_AREA) # average pool
        ref=ref+image
        
        # eeg
        image = edf.get_data()[[index_eeg, index_eeg2],:]
        d0=image.shape[0] #should be d0 = 2
        image.sort(axis=1) 
        image=cv2.resize(image,(size,d0),interpolation=cv2.INTER_AREA) # average pool
        ref_eeg=ref_eeg+image
        
        
        num += 1
        id_all_1000.append(the_id)

print('8c number:', num)
ref=ref/float(num)
np.save(path1+'ref555_shhs_8c', ref)

print('eeg number:',num)
ref_eeg=ref_eeg/float(num)
np.save(path1+'ref555_shhs_eeg', ref_eeg)

out_f=open('./ids_shhs1_1000.txt','w')
out_f.write('\n'.join(id_all_1000))
out_f.close()

