#!/usr/bin/env python
import os, sys
import numpy as np
import pandas as pd
import scipy.io
from glob import glob
import unet as unet
import pickle
from keras.utils.np_utils import to_categorical
from keras import backend as K
import tensorflow as tf
import keras
import argparse
from keras.backend.tensorflow_backend import set_session

from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import sklearn.metrics as metrics

# set fraction of GPU the program can use
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

# set the value of data format convention
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

def evaluations_matrix(gs, pred):
    fpr, tpr, thresholds = metrics.roc_curve(gs, pred, pos_label=1)
    auroc = metrics.auc(fpr, tpr)
    auprc = metrics.average_precision_score(gs, pred)
    return auroc, auprc

def evaluations_matrix_binary(gs, pred):
    """
    Params
    ------
    gs: gold standards
    pred: predicted binary labels 

    Yields
    ------
    sensitivity
    specificity
    precision
    F1-score
    auprc_baseline: P/N in all samples
    """
    conf_mat = confusion_matrix(gs, pred)
    tn, fp, fn, tp = conf_mat.ravel()
    auprc_baseline = (fn+tp)/(tn+fp+fn+tp)
    sensitivity = tp/(tp+fn) # recall
    specificity =tn/(tn+fp)
    precision = tp/(tp+fp)
    accuracy = (tn+tp)/(tn+fp+fn+tp)
    F1= 2*(sensitivity*precision)/(sensitivity+precision)
    return conf_mat, sensitivity, specificity, precision, accuracy, F1, auprc_baseline

def predict(model_path, spec_path, channel_idx, label_idx, path1, path2, test_path):
    # initisate keras
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.95 
    set_session(tf.Session(config=config))
    K.set_image_data_format('channels_last') 

    # hyperparameter
    size= 4096*256
    channel=len(channel_idx)
    print(channel)
    labels = ['Undefined', 'Wake', 'N1', 'N2', 'N3', 'REM', 'Arousal', 'Apnea'] 
    labels = [labels[i] for i in label_idx]
    print(labels)
    batch = 1 
    num_class = len(labels) #arousal 

    allpath = glob(model_path+'*.h5')
    print(allpath)
    model = {i:unet.get_unet(size, channel,num_class) for i in range(len(allpath))}
    for i in range(len(allpath)):
        model[i].load_weights(allpath[i])
     
    # evaluation
    all_test_eva = open('./eva_test.txt', 'w')
    all_test_eva.write('id\tlabel\tAUROC\tAUPRC\tSensitivity\tSpecificity\tPrecision\tAccuracy\tF1\tauprc_baseline\ttotal_length\n')

    pred_all = []
    gs_all = []
    #sleep_conf_mat = np.zeros((5,5)) 
    arousal_conf_mat = np.zeros((2,2))
    apnea_conf_mat = np.zeros((2,2))
    # load files
    all_test_files=open(test_path,'r')
    for filename in all_test_files:
        filename=filename.strip()
        tmp=filename.split('/')
        the_id=tmp[-1]
        print(the_id)

        # select 3  signal channels
        signal = np.load(path1 + the_id + '.npy')[channel_idx,:]
        label = np.load(path2 + the_id + '.npy')[label_idx, :] 
        #print('label shape:', label.shape)
        #d2 = label.shape[1]

        input_pred=np.reshape(signal.T,(batch,size,channel)) 
        output1 = []
        output2 = []
        output3 = []
        # use model to predict, 5 predicts/model then use average as output
        for m in model.values():#[model0, model1, model2, model3, model4]:
            output_ori = m.predict(input_pred)
            output1.append(output_ori[0])
            output2.append(output_ori[1])
            output3.append(output_ori[2])
        output1 = np.mean(np.vstack(output1), axis = 0).T
        output2 = np.mean(np.vstack(output2), axis = 0).T
        output3 = np.mean(np.vstack(output3), axis = 0).T

        print(output1.shape, output2.shape, output3.shape)
        np.save(spec_path+the_id+'_sleep', output1)
        np.save(spec_path+the_id+'_arousal', output2)
        np.save(spec_path+the_id+'_apnea', output3)

        output1_cat = np.round(output1/output1.max(axis =0))

        for i in [1,2,3,4,5]:
            gs_i =label[i,:]
            idx = np.argwhere(gs_i >-1).flatten()
            gs_i = gs_i[idx]
            pred_i = output1[i,:]
            pred_i= pred_i[idx]
            pred_bi = output1_cat[i,:]
            pred_bi = pred_bi[idx] 
            auroc,auprc = evaluations_matrix(gs_i, pred_i)
            try:
                _, sensitivity, specificity, precision, accuracy, F1, auprc_baseline = evaluations_matrix_binary(gs_i, pred_bi)
                all_test_eva.write('%s\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%d\n' % (the_id, labels[i], auroc, auprc, sensitivity, specificity, precision, accuracy, F1, auprc_baseline, len(idx)))
                print('%s\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%d\n' % (the_id, labels[i], auroc, auprc, sensitivity, specificity, precision, accuracy, F1, auprc_baseline, len(idx)))
            except:
                print(the_id, labels[i])
        
        # confusion matrix
        gs_all.extend(label[:6,idx].argmax(axis = 0))
        pred_all.extend(output1_cat[:, idx].argmax(axis = 0))

        gs_i =label[6,:]
        pred_i = output2[0,:]
        # masked
        idx = np.argwhere(gs_i >-1).flatten()
        gs_i = gs_i[idx]
        pred_i = pred_i[idx]
        auroc,auprc = evaluations_matrix(gs_i, pred_i)
        cut = 0.5
        pred_bi = np.where(pred_i>cut, 1, 0)
        try:
            conf_mat, sensitivity, specificity, precision, accuracy, F1, auprc_baseline = evaluations_matrix_binary(gs_i, pred_bi)
            all_test_eva.write('%s\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%d\n' % (the_id, 'Arousal', auroc, auprc, sensitivity, specificity, precision, accuracy, F1, auprc_baseline, len(idx)))
            print('%s\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%d\n' % (the_id, 'Arousal', auroc, auprc, sensitivity, specificity, precision, accuracy, F1, auprc_baseline, len(idx)))
            arousal_conf_mat = arousal_conf_mat+conf_mat
        except:
            print(the_id, 'Arousal', gs_i,pred_bi)

        gs_i =label[7,:]
        pred_i = output3[0,:]
        # masked
        idx = np.argwhere(gs_i >-1).flatten()
        gs_i = gs_i[idx]
        pred_i = pred_i[idx]
        auroc,auprc = evaluations_matrix(gs_i, pred_i)
        cut = 0.5
        pred_bi = np.where(pred_i>cut, 1, 0)
        try:
            conf_mat, sensitivity, specificity, precision, accuracy, F1, auprc_baseline = evaluations_matrix_binary(gs_i, pred_bi)
            all_test_eva.write('%s\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%d\n' % (the_id, 'Apnea', auroc, auprc, sensitivity, specificity, precision, accuracy, F1, auprc_baseline, len(idx)))
            print('%s\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%d\n' % (the_id, 'Apnea', auroc, auprc, sensitivity, specificity, precision, accuracy, F1, auprc_baseline, len(idx)))
            apnea_conf_mat = apnea_conf_mat+conf_mat
        except:
            print(the_id, 'Arousal', gs_i,pred_bi)

    all_test_eva.close()
    df = pd.read_csv('./eva_test.txt', sep = '\t', header=0)
    print(df.groupby('label').mean())

    gs_all = np.array(gs_all)
    pred_all = np.array(pred_all)
    np.save('sleep_gs_all', gs_all)
    np.save('sleep_pred_all', pred_all)
    sleep_conf_mat = confusion_matrix(gs_all, pred_all)
    np.savetxt('sleep_stages_conf_mat.txt', sleep_conf_mat) 
    np.savetxt('arousal_conf_mat.txt', arousal_conf_mat)
    np.savetxt('apnea_conf_mat.txt', apnea_conf_mat)


def main():
        parser = argparse.ArgumentParser(description = 'sleep model test program',
            usage = 'use "python %(prog)s --help" for more information',
            formatter_class=argparse.RawTextHelpFormatter)
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
    
        parser.add_argument('--test_path', type=str, default = 'whole_test_20.txt',
            help = 'test path, default = whole_test_20.txt')

        args = parser.parse_args()
        opts = vars(args)

        run(**opts)


def run(channel_idx, label_idx, sig_path, lab_path, test_path):
        
        model_path = './weights/'
        print('Start predicting ...')
        spec_path = './specs/'
        os.makedirs(spec_path, exist_ok = True)
        predict(model_path, spec_path, channel_idx, label_idx, sig_path, lab_path, test_path)


if __name__ =='__main__':
    main()
