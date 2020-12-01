import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
min_max_scaler = StandardScaler()

def add_RUL(data, factor = 0, piecewise=True, Rearly=130):
    """
    This function appends a RUL column to the df by means of a linear function.
    copyright: https://www.kaggle.com/vinayak123tyagi/damage-propagation-modeling-for-aircraft-engine
    """
    df = data.copy()
    fd_RUL = df.groupby('id')['cycles'].max().reset_index()
    fd_RUL = pd.DataFrame(fd_RUL)
    fd_RUL.columns = ['id','max']
    df = df.merge(fd_RUL, on=['id'], how='left')
    RUL = df['max'] - df['cycles']
    if piecewise:
        # rectify training RUL labels (Rearly = 125)
        idx = RUL > Rearly
        df['RUL'] = RUL
        df['RUL'][idx] = Rearly
    else:
        df['RUL'] = RUL
    df.drop(columns=['max'],inplace = True)
    return df[df['cycles'] > factor]

def sliding_window(sequence, window_size, predict=False):
    """
    This function splits the multivariate timeseries into sliding windows.
    copyright: https://machinelearningmastery.com/how-to-develop-convolutional-neural-network-models-for-time-series-forecasting/
    """
    X = []
    y = []
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + window_size
        # check if we are beyond the dataset
        if end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix, :-1], sequence[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    X = np.array(X)
    y = np.array(y)
    if not predict:
        # randomly shuffle the windows between themselves in unison with the correct labels
        rng_state = np.random.get_state()
        np.random.shuffle(X)
        np.random.set_state(rng_state)
        np.random.shuffle(y)
    return X, y

def test_sliding_window(sequence, window_size):
    # sequence is the array of values associated with ONE engine unit
    X = []
    # find the start of this pattern
    start_ix = len(sequence) - window_size
    seq_x = sequence[start_ix:len(sequence), :]
    X.append(seq_x)
    X = np.array(X)
    return X

# helper class to keep DataFrames structured
class DataFrames:
    def __init__(self, path, split, Rearly=130):
        self.columns = columns = ['id','cycles','altitude','MachNo','TRA','s1','s2','s3','s4','s5','s6','s7','s8',
                                  's9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s26','s20','s21']
        self.split = split
        self.path = path
        self.Rearly = Rearly
        for i in range (1, 5):
            vars(self)[f'{split}_FD00{i}'] = pd.read_csv(f'{path}\\{split}_FD00{i}.txt', header=None, sep=' ')
            # clean the data sets: columns 26, 27 are NaN.
            vars(self)[f'{split}_FD00{i}'].drop(columns=[26, 27], inplace=True)
            vars(self)[f'{split}_FD00{i}'].columns = self.columns
            if self.split=='train':
                vars(self)[f'{self.split}_FD00{i}'] = add_RUL(vars(self)[f'{self.split}_FD00{i}'], Rearly=self.Rearly)
            for column in self.columns:
                if (column != 'id' and column!='RUL'):
                    original = vars(self)[f'{split}_FD00{i}'][[column]].values.astype('float')
                    scaled = min_max_scaler.fit_transform(original)
                    vars(self)[f'{split}_FD00{i}'][column] = scaled
            # vars(self)[f'{split}_FD00{i}'].drop(columns=['s1','s5','s6','s10','s16','s18','s19'], inplace=True)
            
    def get_windowed_dataset(self, engine, window_size, predict):
        if self.split=='train':
            df = vars(self)[f'{self.split}_FD00{engine}']
            array = df.to_numpy()
            X = np.empty((1, window_size, 26))
            y = np.empty((1,))
            for unit in df['id'].unique():
                idx = array[:,0] == unit
                X_unit, y_unit = sliding_window(array[idx], window_size, predict=predict)
                X = np.concatenate((X, X_unit), axis=0)
                y = np.concatenate((y, y_unit), axis=0)
            y = y[1:]
            
        else:
            df = vars(self)[f'{self.split}_FD00{engine}']
            test_RUL = pd.read_csv(f'{self.path}\\RUL_FD00{engine}.txt', header=None, dtype='float')
            y = test_RUL.to_numpy()
            idx = y > self.Rearly
            y[idx] = self.Rearly
            y = y.reshape(len(y),)
            X = np.empty((1, window_size, 26))
            for unit in df['id'].unique():
                array = df.to_numpy()
                idx = array[:,0] == unit
                X_unit = test_sliding_window(array[idx], window_size)
                X = np.concatenate((X, X_unit), axis=0)
                                
        X = X[1:,:,1:] # remove id from training data
        return X, y