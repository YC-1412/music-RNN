# music-RNN
This project generates music based on Chopin's music using LSTM, and scrapes midi files online. 

## Required packages
```
pip install --upgrade music21
pip install tensorflow==2.2
```

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