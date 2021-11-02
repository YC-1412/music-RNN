# music-RNN
This project generates music based on Chopin's music using LSTM. 

Sample out put:
https://user-images.githubusercontent.com/54824364/139856754-2de56a95-f7ef-483c-9665-9b0cc7e02beb.mp4

## Instruction
1. Clone this repository 
2. Run "music_simple.ipynb"

## Required packages
```
pip install --upgrade music21
pip install tensorflow==2.2
```

## Result
- A trained model "model-chopin_100.h5" in the [model folder](https://github.com/YC-1412/music_RNN/tree/main/models)
- Generated music midi file "test_output_100.mid" and it's pdf in the [output folder](https://github.com/YC-1412/music_RNN/tree/main/output)

## Data
95 midi files of Chopin's music scraped from https://www.midiworld.com/chopin.htm
A processed and cleaned data file `chopin_clean` is uploaded to this repository. The notebook can be run by simply changing the filepath.

## Copyright
Chopin's music is public accessible. Please contact me at: yuricao1412@gmail.com if scraping is not allowed.

This project is inspired by [this blog](https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5).

## Organization of this directory
```
.
│  music_simple.ipynb
│  networks.py
│  README.md
│  utils.py
│      
├─data
│  └─chopin
│          chopin_clean
│          
├─models
│      model-chopin_100.h5
│      
└─output
        chopin_100.pdf
        test_output_100.mid
```
