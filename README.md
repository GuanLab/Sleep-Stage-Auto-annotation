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

Since the lengths of sleep recordings are different, we first make uniform these recordings to the same 8-million length (2^23 = 8,388,608) by padding zeros at both the beginning and the end. 

```
unzip ref555.zip
python uniform_8m.py
```
Then we separate the 994 records into a training set and a test set and quantile normalize the training data.

### 3. model training
The model we present here used 11 channels to train the model. You can adjust the number of channels you would like to use.

Run the following command to train the model. Note that i is required, which specifies how many models you would like to train. You can just set it to 1. 
```
python3 train.py i 
```
You can use the following command if you want to keep track of the training and validation loss.
```
python3 train.py i | tee -a log_i.txt
```

### 4. prediction and scoring

Now you can run predictions using the following command:
```
python predict.py
```
It will generate a file for each record called "record_name.vec", each line corresponds to the prediction for each time point in the original polysomnogram.

The AUPRC and AUROC are also calculated in this process. They will be write into a file documenting ROC and AUPRC for each stage in each record.


