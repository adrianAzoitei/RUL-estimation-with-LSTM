from models.LSTM import build_model
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler, EarlyStopping
import numpy as np

def train(X, y, ckpt_path, log_dir, sequence_length, train=True, config=[32, 64, 8, 8]):
    checkpoint = ModelCheckpoint(ckpt_path, monitor='val_root_mean_squared_error', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
    tensorboard_callback = TensorBoard(log_dir=log_dir)
    earlystop = EarlyStopping(monitor='val_root_mean_squared_error', min_delta=0, patience=15, verbose=0, mode='min')
    n_features = len(X[1,1,:])

    # build model
    model = build_model(sequence_length, n_features, config)
    if train:
        history = model.fit(X,
                            y,
                            validation_split=0.2,
                            epochs=300,
                            batch_size=200,
                            verbose=2,
                            shuffle=True,
                            callbacks=[checkpoint,
                                       earlystop, 
                                       tensorboard_callback])
        return model, history
    else:
        model.load_weights(ckpt_path)
        return model

    