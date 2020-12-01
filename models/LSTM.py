import tensorflow as tf
from keras import backend as K
from keras.optimizers import RMSprop
from keras.regularizers import l2
import numpy as np

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

#Error Function for Competitive Data
def score(y_true, y_pred, a1=10, a2=13):
        score = 0.0
        d = y_pred - y_true
        for i in d:
                if i >= 0 :
                        res = tf.math.exp(i/a2) - 1
                        score += res[0]   
                else:
                        res = tf.math.exp(-i/a1) - 1
                        score += res[0]
        return score

opt = RMSprop(learning_rate=0.001)
def build_model(sequence_length, n_features, config):
        model = tf.keras.models.Sequential([
                tf.keras.layers.LSTM(config[0], 
                                     input_shape=(sequence_length, n_features),
                                     return_sequences=True,
                                     kernel_regularizer=l2(0.01),
                                     bias_regularizer=l2(0.01)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(config[1],
                                     kernel_regularizer=l2(0.01), 
                                     return_sequences=False,
                                     bias_regularizer=l2(0.01)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(config[2], kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
                tf.keras.layers.Dense(config[3], kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
                tf.keras.layers.Dense(1),
                tf.keras.layers.Activation('linear')])
        model.compile(optimizer=opt,
                      loss='mse',
                      metrics=[root_mean_squared_error, score])
        model.summary()
        return model