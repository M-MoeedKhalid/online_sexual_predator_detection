import numpy as np
import tensorflow as tf
from keras import Sequential, Input
from keras.layers import LSTM, Dense, Embedding, CuDNNLSTM

import os


def train_rnn(X, y, max_len=None):
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/cpu:0", "/gpu:0"])
    if not max_len:
        max_len = 100  # change this back to 100
    input = Input(shape=(X.size, max_len), dtype='float64')
    word_input = Input(shape=(max_len,), dtype='float64')

    model = Sequential()
    model.add(Embedding(input_dim=X.size, output_dim=4, input_length=max_len))
    # model.input(X)
    model.add(CuDNNLSTM(2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.metrics.AUC()])
    print(model.summary())

    num_epochs = 10
    model.fit(X, y, epochs=num_epochs, validation_data=(X, y), verbose=1)

    # model.save_weights()
