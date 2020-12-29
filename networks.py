import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, BatchNormalization, Bidirectional, Concatenate, RepeatVector
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from utils import generate_midi, generate_notes


class LSTM_model():
    '''
    Build and  train RNN model
    Modified from https://github.com/Skuldur/Classical-Piano-Composer/blob/master/lstm.py
    '''
    def __init__(self, input_shape, output_shape, model_path):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = None
        self.model_path = model_path  # path where the model will be saved
        self.model_name = ""
        
    def create_model(self):
        '''
        Create RNN model
        '''
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
        '''
        Train model and save checkpoints
        Args:
            @network_input: input training data
            @network_output: output training data
            @epochs: training epochs
            @batch_size: batch size
        '''
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
        '''
        Generate music using trained model
        Args:
            @network_input: inpout matrix of RNN
            @pitchnames: a set of all notes in the training music
            @length: # nodes in generated music
            @speed: speed of generated music. If None, speed=0.5 (0.5s per note). If a number, will generate random speed from (0, number)
        Return:
            prediction_output: list of string. A note sequence
            output_notes: music21 readable midi format music
        Call:
            generate_notes(network_input, pitchnames)
            generate_midi(prediction_output, speed=None)
        '''
        # check if a model is loaded
        if self.model == None:
            print("Please train a model first using '.train'")
            return
        
        # get a random sequence to start training
        pattern, int_to_note, n_vocab = generate_notes(network_input, pitchnames)
        prediction_output = []
        
        # generate 500 notes
        for note_index in range(length):
            prediction_input = np.reshape(pattern, (1, len(pattern), 1))
            # prediction is a matrix showing the probability of each note
            # index is the index of note with max prob
            prediction = self.model.predict(prediction_input, verbose=0)
            index = np.argmax(prediction)
            # map index to note name
            result = int_to_note[index]
            prediction_output.append(result)
            # standardize the new note to make it consistent with the training input
            pattern = np.append(pattern[1:],index/n_vocab)
        # get music21 format midi files with certain or random speed given a note sequence
        output_notes = generate_midi(prediction_output, speed=speed)

        return prediction_output, output_notes

    def load_model(self, model_name, mode='colab'):
        '''
        Load trainned model to resume training or generate music
        Args:
            @model_name: name of the saved model
            @mode: "colab" or "jupyter", indicating whether the saved model is trained on Google Colab or GCP
        '''
        if mode == 'colab':
            self.create_model()
            self.model.load_weights(self.model_path+model_name)
        elif mode == 'jupyter':
            self.model = tf.keras.models.load_model(self.model_path+model_name)
        print('Successfully loaded model!')