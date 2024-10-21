# DAN Sentiment Classifier

## Overview

This project is for UCSD CSE256 PA1, contains codes that implements a Deep Averaging Network (DAN) model for sentiment classification with Byte-Pair Encoding (BPE) and GloVe word embeddings.

## Code Structures
Beside the code provided by default, `DANmodels.py` contains codes that implements DAN model, as well as `SentimentDatasetDAN` and `SentimentDatasetBPE` class which is for creating dataset class for DAN (word token) and DAN with BPE tokenization.

`main.py` updated accordingly, with argument parser that takes training/model parameters into account.

## How To Run
### 1a (DAN with GloVe embeddings)
```bash
python main.py --model DAN --embedding_size 50
python main.py --model DAN --embedding_size 300
```
Above code will run DAN model with 50d and 300d pre-trained GloVe embeddings.

Important parameters are printed. Also, plot will be saved in plots folder.
Some example output (partial) looks like below.
```bash
Read in 14923 vectors of size 300
Data loaded in : 0.13616323471069336 seconds
Training DAN model:
Epoch #10: train accuracy 0.748, dev accuracy 0.763
Epoch #20: train accuracy 0.755, dev accuracy 0.794
Epoch #30: train accuracy 0.755, dev accuracy 0.796
Epoch #40: train accuracy 0.760, dev accuracy 0.778
Epoch #50: train accuracy 0.757, dev accuracy 0.768
Epoch #60: train accuracy 0.767, dev accuracy 0.788
Epoch #70: train accuracy 0.774, dev accuracy 0.779
Epoch #80: train accuracy 0.772, dev accuracy 0.786
Epoch #90: train accuracy 0.775, dev accuracy 0.800
Epoch #100: train accuracy 0.775, dev accuracy 0.794
Trained in : 113.01143074035645 seconds


Training accuracy plot saved as plot_gl_word_embdim300_hid128_lr0.001.png
```

### 1b (DAN with random embeddings)
You can run following (recommend embedding_size =50 to faster result), also for embedding_size, since I used same indexer as the GloVe, currently only 50d and 300d available for random embedding. Embedding dimension is flexible for BPE.
```bash
python main.py --model DAN --embedding_size 50 --random_embedding --lr 0.0001
```
