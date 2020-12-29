import os
import zipfile
import urllib.request as url
import pandas as pd
import numpy as np
import pickle
from music21 import converter, instrument, note, chord
import random



def download_data(path):
    '''
    Download the midi data from the piano-midi website, which is approximately 300KB.
    The data (a .tar.gz file) will be store in the given path.
    This is not used for music_simple, which uses data scraped from https://www.midiworld.com/chopin.htm
    Args:
        @path: path where the data saved/to be saved
    '''
    # If the path doesn't exist, create it first
    if not os.path.exists(path):
        os.mkdir(path)
    # If the data hasn't been downloaded yet, download it
    if not os.path.exists(path+'midi_data.zip'):
        print('Start downloading midi_data...')
        url.urlretrieve("http://www.piano-midi.de/zip/chopin.zip",
                        path+"midi_data.zip")
        print('Download complete.')
    else:
        print('midi_data.zip already exists.')
        
def unpack_data(path, zipfile_name):
    """
    Unpack the zipfile. The unpacked data will be store in the given path.
    Args:
        @path: path where the data saved/to be saved
        @zipfile_name: name of the zip file
    """
    # If the data hasn't been downloaded yet, download it first
    if not os.path.exists(path+zipfile_name+'.zip'):
        if zipfile_name == 'midi_data':
            download_data(path)
        else:
            print(zipfile_name+'.zip not exist. Please download first.')
        
    else:
        print(path+zipfile_name+'.zip already exists')
    # Check if the package has been unpacked, otherwise unpack the package
    if not os.path.exists(path+zipfile_name):
        os.mkdir(path+zipfile_name)
        print('Begin extracting...')
        with zipfile.ZipFile(path+zipfile_name+'.zip', 'r') as zip_ref:
            zip_ref.extractall(path+zipfile_name)
        print('Unzip complete.')
    else:
        print(path+zipfile_name+'.zip already unzipped.')
        

        
def process_music(path, clean_name):
    '''
    Load and transform midi data to matrix 
    Modified from https://github.com/Skuldur/Classical-Piano-Composer/blob/master/lstm.py
    Args:
        @path: path of the file to be processed
        @clean_name: name of the clean file to be saved
    Return:
        notes: a list of string. Cleaned notes
    '''
    
    print('Start loading midi files...')
    notes = []
    for fname in os.listdir(path):
        if fname[-4:] not in ('.mid','.MID'):
            continue
            
        file_path = path+fname
        midi = converter.parse(file_path)
        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            # transform music21 format to string
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
        
        print("Loaded {}".format(fname))
        
    # save clean file
    with open(path+clean_name, 'wb') as filepath:
        pickle.dump(notes, filepath)
    
    print('Finish loading, cleaned file saved as {path}/{clean_name}'.format(path=path, clean_name=clean_name))
        
    return notes
    

    
def prepare_sequences(notes):
    '''
    Prepare the sequences used by the Neural Network. Transform notes to matrix
    Modified from https://github.com/Skuldur/Classical-Piano-Composer/blob/master/lstm.py
    Args:
        @notes: a list of string. Notes of input music
    Return:
        network_input: inpout matrix of RNN
        network_output: output matrix of RNN
        pitchnames: a set of all notes in the training music
    '''
    sequence_length = 100

    # get all pitch names
    pitchnames = sorted(set(notes))
    n_vocab = len(pitchnames)

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)
    # generate output matrix
    network_output = np.array(pd.get_dummies(network_output))

    return (network_input, network_output, pitchnames)
    

def generate_notes(network_input, pitchnames):
    '''
    Pick a random sequence from the input as a starting point for the prediction 
    Modified from https://github.com/Skuldur/Classical-Piano-Composer/blob/master/predict.py
    Args:
        @network_input: inpout matrix of RNN
        @pitchnames: a set of all notes in the training music
    Return:
        pattern: a random sequence from the training music
        int_to_note: a dictionary for mapping network_input to note names
        n_vocab: length of pitchnames, indicating the number of unique notes in the training music. Used for normalizing data
    '''
    
    start = np.random.randint(0, network_input.shape[0]-1)
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    # pick a random sequence from the input as a starting point for the prediction
    pattern = network_input[start]
    n_vocab = len(pitchnames)
    return pattern, int_to_note, n_vocab


def generate_midi(prediction_output, speed=None):
    '''
    Generate music21 format midi files with certain or random speed given a note sequence
    Modified from https://github.com/Skuldur/Classical-Piano-Composer/blob/master/predict.py
    Args:
        @prediction_output: list of string. A note sequence
        @speed: speed of generated music. If None, speed=0.5 (0.5s per note). If a number, will generate random speed from (0, number)
    Return:
        output_notes: music21 readable midi format music
    '''
    offset = 0
    output_notes = []
    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord, split
        if ('.' in pattern) or pattern.isdigit():
            index_in_chord = pattern.split('.')
            notes = []
            for note_index in index_in_chord:
                # if note_index is a number
                if note_index.isdigit():
                    note_index = int(note_index)
                new_node = note.Note(note_index)
                new_node.storedInstrument = instrument.Piano()
                notes.append(new_node)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_node = note.Note(pattern)
            new_node.storedInstrument = instrument.Piano()
            new_node.offset = offset
            output_notes.append(new_node)
        # increase offset each iteration so that notes do not stack
        if speed == None:
            offset += 0.5
        else:
            offset += round(random.uniform(0, speed),2)
        
    return output_notes