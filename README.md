## Sleep stage auto-annotation based on polysomnographic records

## Installation
Git clone a copy of code:
```
git clone https://github.com/GuanLab/Sleep-Stage-Auto-annotation.git
```
## Required dependencies

* [python](https://www.python.org) (3.6.5)
* [numpy](http://www.numpy.org/) (1.13.3). 
* [tensorflow](https://www.tensorflow.org/) (1.8.0)
* [keras](https://keras.io/) (2.0.4)

## Dataset

We used the public training data from the 2018 PhysioNet challenge, which contains 994 polysomnographic recordings. Details and download link can be found [HERE](https://physionet.org/physiobank/database/challenge/2018/).

The Sleep Heart Health Study dataset is also used to validate the model. Details about the dataset and data request link can be found [HERE](https://sleepdata.org/datasets/shhs).

## Model development 

### 1. prepare polysomnogram data (borrowed from DeepSleep repo)

First download the data and put them into the folder "./data/training/". The training dataset is approximately 135 GB in size.


### 2. preprocessing

The code used for data preprocess can be found in `./preprocess` folder. Both PhysioNet and SHHS data were preprocessed in the same fashion, while details may be different since the two datasets were stored in different formats.
The data preprocess is carried out in following steps:

* 1.  split train and test sets: 

  run `python split_train_test.py`: this will generate file ids for both training set and test set, stored in `id_train80.txt` and  `id_test20.txt` 

* 2. create reference distribution for input signal:

  run `python create_ref_shhs.py` : this will generate `ref555_shhs_8c.npy` and `ref555_shhs_eeg.npy` stored as numpy array. The reference distribution will be used in quantile normalzation in the next step.

* 3. signal preprocess and quantile normalization:

  run `python process_shhs_signal_label.py`: this will generate both the processed 8 channel signals and eeg signals (2 channels). Also this will generate the corresponding prediction labels. Of note, since the lengths of sleep recordings are different, we first make uniform these recordings to the same 8-million length (2^23 = 8,388,608) by padding zeros at both the beginning and the end. The labels and signals should be padded in the same locations and avoid mismatch.

### 3. model training
The model we present here used 11 channels to train the model. You can adjust the number of channels you would like to use.

Run the following command to train the model. Note that i is required, which specifies how many models you would like to train. You can just set it to 1. 
```
python train.py i
```

### 4. prediction and scoring

Now you can run predictions using the following command:
```
python predict.py
```
It will generate a file for each record called "record_name.vec", each line corresponds to the prediction for each time point in the original polysomnogram.

The AUPRC and AUROC are also calculated in this process. They will be write into a file documenting ROC and AUPRC for each stage in each record.


