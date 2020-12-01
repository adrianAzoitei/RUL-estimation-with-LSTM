import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import math
import matplotlib.pyplot as plt
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
    print(f'Inferring on FD00{i}')

    # take sequence length of the shortest test trajectory in each subset
    if i == 1 or i == 3:
        sequence_length = 30
    elif i == 2:
        sequence_length = 20
    elif i == 4:
        sequence_length = 15

    # load training data
    traindata  = DataFrames(DATA_DIR, 'train')
    train_X, train_y = traindata.get_windowed_dataset(i, sequence_length, predict=True)
    # load test data
    testdata  = DataFrames(DATA_DIR, 'test')
    test_X, test_y = testdata.get_windowed_dataset(i, sequence_length, predict=True)
    n_features = len(train_X[1,1,:])

    # inspect one of the dataframes
    print(traindata.train_FD001)

    # checkpoints and logs for TensorBoard
    ckpt_file ="weights_FD00{}.h5".format(i)
    ckpt_path = os.path.join(CKPT_DIR, ckpt_file)
    logdir = os.path.join(LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    # load model
    config = [
        32, # units of first LSTM layer
        32, # units of second LSTM layer
        8,  # units of first Dense layer
        8   # units of second Dense layer
    ]
    model = train(train_X, train_y, ckpt_path, logdir, sequence_length, train=False, config=config)

    # run some predictions on training set
    predictions = model.predict(train_X)
    for j in range(10):
        print(f'Ground truth vs prediction on train data:{train_y[j]} - {predictions[j]}')
    
    predictions = predictions.reshape(len(predictions),)
    trainRMSE = math.sqrt(sum((predictions - train_y) ** 2)/len(train_y))
    print('Train set RMSE:{}'.format(trainRMSE))
    unit = np.arange(0, len(train_y))
    plt.style.use(['ggplot'])
    plt.figure(1,figsize=(7,5))
    plt.plot(unit[0:300], predictions[0:300],'r--', label='Predictions')
    plt.plot(unit[0:300], train_y[0:300], 'b-', label='Ground truth')
    plt.xlabel(f'Training cycles FD00{i}')
    plt.ylabel('RUL')
    plt.title('Train RMSE: {:.4}'.format(trainRMSE))
    plt.legend()
    plt.grid(True)
    
    # run some predictions on test set
    predictions = model.predict(test_X)
    for jj in range(10):
        print(f'Ground truth vs prediction on test data:{test_y[jj]} - {predictions[jj]}')

    # get test set RMSE
    predictions = predictions.reshape(len(predictions),)
    testRMSE = math.sqrt(sum((predictions - test_y) ** 2)/len(test_y))
    print(f'Test set RMSE:{testRMSE}')

    # plot test units predictions
    unit = np.arange(0, len(test_y))
    unit = unit.reshape(len(unit),1)
    test_y = test_y.reshape(len(test_y),1)
    predictions = predictions.reshape(len(predictions),1)
    compare = np.concatenate((test_y, predictions), axis=1)
    compare = np.sort(compare, axis=0)
    plt.figure(2,figsize=(7,5))
    plt.plot(unit, compare[:,1], 'r--', label='Predictions')
    plt.plot(unit, compare[:,0], 'b-', label='Ground truth')
    plt.xlabel('Test units FD00{}'.format(i))
    plt.ylabel('RUL')
    plt.title('Test RMSE: {:.4}'.format(testRMSE))
    plt.legend()
    plt.grid(True)
    plt.show()