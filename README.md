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
Important parameters are printed. For example,
```bash
Read in 14923 vectors of size 300
Data loaded in : 0.13616323471069336 seconds
Training DAN model with the following parameters:
  Hidden size: 128
  Dropout (hidden layers): 0.5
  Number of layers: 2
  Word dropout: 0.4
  Fine-tune embeddings: False
  Random embeddings: False
  Batch size: 16
  Epochs: 100
  Learning rate: 0.001
  Weight decay: 1e-05
  Max length: 100

Training DAN model:
Epoch #10: train accuracy 0.748, dev accuracy 0.763
```
