## parse annotation
import xml.etree.ElementTree as ET
import mne
import numpy as np
import sys
import os
import scipy.io
import cv2
import tqdm

def anchor (ref, ori): # input m*n np array
    d0=ori.shape[0]
    d1=ori.shape[1]
    ref=cv2.resize(ref,(d1,d0),interpolation=cv2.INTER_AREA)
    ori_new=ori.copy()
    for i in range(d0):
        ori_new[i,np.argsort(ori[i,:])]=ref[i,:]
    return ori_new

path1='/nfs/disk3/hyangl/2018/physionet/data/shhs/'

# output path
path2 = './shhs_image_avg5/'
path2_eeg='./shhs_image_eeg_avg5_noanchor/'
path3='./shhs_label_avg5/'
os.system('mkdir -p ' + path2)
os.system('mkdir -p ' + path2_eeg)
os.system('mkdir -p ' + path3)

size= 2**20 # 1048576
freq=125
norm_freq = 125
scale_pool = 5 # pool by 5 because sampled at 125 Hz, while PhysioNet sampled at 200Hz
channel_num = 8 # input 8 channels
channel_num_eeg = 2 # eeg only
class_num = 8 # label class

# read ids
id_all=[]
f=open('./ids_shhs1_1000.txt','r')
for line in f:
    id_all.append(line.rstrip())
f.close()

ref555=np.load('./ref555_shhs.npy')
ref555_eeg=np.load('./ref555_shhs_eeg.npy')

i = 0
while i < len(id_all):
    the_id = id_all[i]
    the_id=the_id.rstrip()
    print(the_id,i)
    i+=1
    
    # image
    edf = mne.io.read_raw_edf(path1 + the_id + '.edf', verbose=False)
    
    name_chan = edf.info['ch_names'] # 'SaO2', 'H.R.', 'EEG(sec)', 'ECG', 'EMG', 'EOG(L)', 'EOG(R)', 'EEG', 'THOR RES', 'ABDO RES', 'POSITION', 'LIGHT', 'NEW AIR', 'OX stat'
    # all 8+2 channels
    try:
        index_sao2 = name_chan.index('SaO2')
        index_hr = name_chan.index('H.R.')
        index_ecg = name_chan.index('ECG')
        index_thor = name_chan.index('THOR RES')
        index_abd = name_chan.index('ABDO RES')
        index_posi = name_chan.index('POSITION')
        index_light = name_chan.index('LIGHT')
        index_ox = name_chan.index('OX stat')
        # 2 eeg
        index_eeg = name_chan.index('EEG')
        index_eeg2 = name_chan.index('EEG(sec)')
    except:
        print('channel not found!')
        continue
    else:
        
        # 8 channels
        image_ori = edf.get_data()[[index_sao2, index_hr, index_ecg, index_thor, index_abd, index_posi, index_light, index_ox],:]
        image_ori = cv2.resize(image_ori,(int(edf.n_times/scale_pool),channel_num),interpolation=cv2.INTER_AREA)
        image = anchor(ref555, image_ori)
        d0=image.shape[0]
        d1=image.shape[1]
        if(d1 < size):
            image=np.concatenate((image,np.zeros((d0,size-d1))),axis=1)
        np.save(path2 + the_id , image)
        
        #eeg only
        image_ori = edf.get_data()[[index_eeg, index_eeg2],:]
        image_ori = cv2.resize(image_ori,(int(edf.n_times/scale_pool),channel_num_eeg),interpolation=cv2.INTER_AREA)
        image = image_ori
        image = anchor(ref555_eeg, image_ori)
        d0=image.shape[0]
        d1=image.shape[1]
        if(d1 < size):
            image=np.concatenate((image,np.zeros((d0,size-d1))),axis=1)
        np.save(path2_eeg + the_id , image)
    
    # labels
    d1 =int(edf.n_times/scale_pool)
    label=np.zeros((class_num, size))
    root = ET.parse(path1 + the_id + '-nsrr.xml').getroot()
    iteration_array = np.arange(1,len(root[2]))
    for j in iteration_array:
        #print(root[2][i][1].text)
        annt_text = root[2][j][1].text
        start = int(float(root[2][j][2].text) * float(freq) / float(scale_pool))
        end=start + int(float(root[2][j][3].text) * float(freq) / float(scale_pool))
        # 5 sleep stages
        cat = 0
        if annt_text == 'Wake|0':
            cat=1
        elif annt_text == 'Stage 1 sleep|1':
            cat=2
        elif annt_text == 'Stage 2 sleep|2':
            cat =3
        elif annt_text in ('Stage 3 sleep|3', 'Stage 4 sleep|4'):  # deep sleep stage combined
            cat = 4
        elif annt_text == 'REM sleep|5':
            cat = 5
        elif annt_text == 'Unscored|9':
            cat = 0
        elif annt_text in ('Hypopnea|Hypopnea', 'Obstructive apnea|Obstructive Apnea','Central apnea|Central Apnea','Mixed apnea|Mixed Apnea'):
            cat = 7
        elif annt_text in ('External arousal|Arousal (External Arousal)','Arousal|Arousal (STANDARD)','Arousal|Arousal ()','Arousal|Arousal (Standard)','Arousal resulting from Chin EMG|Arousal (CHESHIRE)','ASDA arousal|Arousal (ASDA)'):
            cat = 6
        elif annt_text in ('Unsure|Unsure','SpO2 artifact|SpO2 artifact', 'SpO2 desaturation|SpO2 desaturation','Respiratory artifact|Respiratory artifact'):
            continue
        else:
            print('Unknown: ', annt_text)
            continue 
        #print(annt_text, cat)
        label[cat,start:end]=1
    # NOTE: make sure if the location are not anntotated by the five stages, annotated as unscored. 
    #label[0, np.argwhere(label[:6,:].sum(axis = 0)==0).flatten()] = 0

    # NOTE: add unscoring part to arousal: wake and apnea/hyponea set to -1(masked), following the same scoring rule as PhysioNet Challenge
    label[6, np.argwhere(label[1,:]==1).flatten()] = -1 # wake
    label[6, np.argwhere(label[7,:]==1).flatten()] = -1 # apnea

    #pad label to size by -1
    label[:, d1:] = -1
    #print(label.shape)
    test = set(np.sum(label[:6,:], axis = 0))
    print(test) # should be larger than 1
    if min(test) == 0:
        print("empty timepoint!")
    np.save(path3 + the_id , label)
    


