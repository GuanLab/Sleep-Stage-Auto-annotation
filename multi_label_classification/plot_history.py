import pickle
from plot_keras_history import plot_history
import matplotlib.pyplot as plt

#import tensorflow as tf
#import keras
#import conv as conv

fold_idx = 'test'

#size = 220500
#channel = 1
#batch = 32
#model = conv.conv(size,channel)
#name_model='weights_' + str(fold_idx) + '.h5'

#model.load_weights(name_model)
def plot(fold_idx):
    history=pickle.load(open('history_'+ str(fold_idx)+'.pkl', 'rb'))
    plot_history(history)
    plt.show()
    plot_history(history, path='standard'+str(fold_idx)+'.png')
    plt.close()


for fold_idx in [1,2,3,4,5]: #range(5):
    try:
        plot(fold_idx)
    except:
        pass
