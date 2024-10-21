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
		Data loaded in : 0.12199974060058594 seconds
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
		Epoch #10: train accuracy 0.750, dev accuracy 0.776
		Epoch #10: train loss 0.506, dev loss 0.44
		...
		Epoch #90: train accuracy 0.781, dev accuracy 0.789
		Epoch #90: train loss 0.456, dev loss 0.445
		Epoch #100: train accuracy 0.776, dev accuracy 0.783
		Epoch #100: train loss 0.455, dev loss 0.452
		Trained in : 90.32789349555969 seconds


Training accuracy plot saved as plot_gl_word_embdim300_hid128_lr0.001.png
```

### 1b (DAN with random embeddings)
You can run following (recommend embedding_size =50 to faster result). I used same indexer as the GloVe, but embedding dimension is flexible since using random initialization.
```bash
python main.py --model DAN --embedding_size 50 --random_embedding --lr 0.0001
```

### 2a (DAN with BPE)
You can run the following code:
```bash
python main.py --model DAN --lr 0.0001 --embedding_size 100 --tokenization bpe --bpe_vocab_size 1000
```
Vocab size must be larger than 256, since initial vocabulary start with UTF-8 encoding.
