import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, BatchNormalization, Bidirectional, Concatenate, RepeatVector
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from utils import generate_midi, generate_notes


class LSTM_model():
    def __init__(self, input_shape, output_shape, model_path):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = None
        self.model_path = model_path
        self.model_name = ""
        
    def create_model(self):
        model = Sequential([
            LSTM(512,
                 input_shape=(self.input_shape[1], self.input_shape[2]),
                 recurrent_dropout=0.3,
                 return_sequences=True
                ),
            LSTM(512, return_sequences=True, recurrent_dropout=0.3,),
            LSTM(512),
            BatchNormalization(),
            Dropout(0.3),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(self.output_shape[1], activation='softmax')
        ])
    
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        self.model = model
    
    def train(self, network_input, network_output, epochs=10, batch_size=128):
        if self.model == None:
            print("Will create a model first using '.create_model'")
            self.create_model()
            
        model_path = self.model_path
        if not os.path.exists(model_path):
            os.mkdir(model_path)
            
        filepath = model_path+"model-"+self.model_name+"-{epoch:02d}-{loss:.4f}.h5"
        checkpoint = ModelCheckpoint(
            filepath,
            monitor='loss',
            verbose=0,
            save_best_only=True,
            mode='min'
        )
        callbacks_list = [checkpoint]

        self.model.fit(network_input, network_output, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)
        
    def generate(self, network_input, pitchnames, length=100, speed=None):
        if self.model == None:
            print("Please train a model first using '.train'")
            return
        
        pattern, int_to_note, n_vocab = generate_notes(network_input, pitchnames)
        prediction_output = []
        
        # generate 500 notes
        for note_index in range(length):
            prediction_input = np.reshape(pattern, (1, len(pattern), 1))

            prediction = self.model.predict(prediction_input, verbose=0)

            index = np.argmax(prediction)
            result = int_to_note[index]
            prediction_output.append(result)

            pattern = np.append(pattern[1:],index/n_vocab)
        
        output_notes = generate_midi(prediction_output, speed=speed)

        return prediction_output, output_notes

    def load_model(self, model_name, mode='colab'):
        if mode == 'colab':
            self.create_model()
            self.model.load_weights(self.model_path+model_name)
        elif mode == 'jupyter':
            self.model = tf.keras.models.load_model(model_path+model_name)
        print('Successfully loaded model!')