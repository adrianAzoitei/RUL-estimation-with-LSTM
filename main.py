import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import math

# (optional) required for local running on GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# import system and file handling packages
import os
import sys
# set path to working directory
path = os.getcwd()
rootpath = os.path.join(path, os.pardir)
roothpath = os.path.abspath(rootpath)
sys.path.insert(0, path)

#import defined variables and methods
from utils import DATA_DIR, CKPT_DIR, LOG_DIR
from data_loader.data_prep import DataFrames
from trainers.trainer import train

# take files sepparately
for i in range(2, 3):
    train_file = 'train_FD00{}.txt'.format(i)
    test_file  = 'test_FD00{}.txt'.format(i)
    RUL_file = 'RUL_FD00{}.txt'.format(i)
    print('Training on FD00{}'.format(i))
    sequence_length = 50

    # load training data
    Rearly = 130 # change the maximum RUL considered
    data = DataFrames(DATA_DIR, 'train', Rearly=Rearly) 
    train_X, train_y = data.get_windowed_dataset(i, sequence_length, predict=False)
    print(data.train_FD001)
    print(train_X[0,0,:])
    print(train_y[:5])
    
    # checkpoints and logs for TensorBoard
    ckpt_file ="weights_FD00{}.h5".format(i)
    ckpt_path = os.path.join(CKPT_DIR, ckpt_file)
    logdir = os.path.join(LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S"))

    # train LSTM
    config = [
        32, # units of first LSTM layer
        32, # units of second LSTM layer
        8,  # units of first Dense layer
        8   # units of second Dense layer
    ]
    model, history = train(train_X, train_y, ckpt_path, logdir, sequence_length, train=True, config=config)
