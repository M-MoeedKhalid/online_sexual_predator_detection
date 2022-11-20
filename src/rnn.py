from datetime import datetime

import numpy as np
import tensorflow as tf
from keras import Sequential, Input
from keras.layers import LSTM, Dense, Embedding, CuDNNLSTM, SimpleRNN

import os


def train_rnn(X, y, max_len=None):
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/cpu:0", "/gpu:0"])
    if not max_len:
        max_len = 100  # change this back to 100
    dt = datetime.now()
    checkpoint_path = f"../training_1/cp-{dt}.ckpt"
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    ## Model
    # model = Sequential()
    # model.add(Embedding(input_dim=X.size, output_dim=1, input_length=max_len))
    # # model.input(X)
    # model.add(CuDNNLSTM(2))
    # model.add(Dense(1, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.metrics.AUC()])

    model = Sequential()
    model.add(Embedding(500, 32))
    model.add(SimpleRNN(32))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.metrics.AUC()])

    num_epochs = 10
    model.fit(X, y, epochs=num_epochs, validation_data=(X, y), verbose=1, callbacks=[cp_callback])

    # model.save_weights()
