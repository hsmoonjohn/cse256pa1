# DANmodels.py

import torch
from torch import nn
import torch.nn.functional as F
from sentiment_data import read_sentiment_examples, read_sentiment_examples_bpe
from torch.utils.data import Dataset
import re
from collections import Counter

# Step 1: Define the helper functions for BPE
def get_stats(ids):
    """
    Get statistics of byte pair frequencies in a list of token ids.
    """
    counts = Counter()
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, new_token):
    """
    Merge a pair of token ids in a list into a new token.
    """
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(new_token)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

def encode_bpe(byte_sequence, merges):
    """
    Encode a byte sequence into BPE tokens using the pre-trained BPE vocabulary and merge rules.
    
    Args:
        byte_sequence (list of int): A list of integers representing the byte-level tokens.
        bpe_vocab (dict): The BPE vocabulary with byte pairs as keys and merged tokens as values.
        merges (dict): A dictionary of merges with pairs of byte tokens as keys and merged token IDs as values.
    
    Returns:
        list of int: The final encoded sequence of BPE tokens.
    """
    tokens = list(byte_sequence)  # Start with the byte sequence (list of integers)
    
    # Apply BPE merges based on the pre-trained bpe_vocab and merge rules
    while len(tokens) >= 2:
        pairs = get_stats(tokens)  # Get all consecutive byte pairs
        
        # Find the pair that exists in the merges dictionary
        pair = min(pairs, key=lambda p: merges.get(p, float("inf")))
        
        # Stop if no more pairs are found in the merges dictionary
        if pair not in merges:
            break
        
        # Merge the pair and replace with the new token from merges
        new_token = merges[pair]
        tokens = merge(tokens, pair, new_token)  # Perform the merge
    
    return tokens  # Return the final encoded BPE token sequence

class DAN(nn.Module):
    def __init__(self, embeddings, hidden_size=300, dropout=0.3, num_layers=4, dropoutword=0.3, fine_tune_embeddings=False, random_embedding=False):
        """
        Initialize the Deep Averaging Network (DAN).
        
        Args:
        - embeddings: Pretrained word embeddings (WordEmbeddings class).
        - hidden_size: Size of the hidden layer.
        - dropout: Dropout probability for regularization.
        - num_layers: Number of hidden layers.
        - fine_tune_embeddings: Whether to fine-tune the pretrained embeddings during training.
        """
        super(DAN, self).__init__()
        
        # Embedding layer
        embedding_dim = embeddings.get_embedding_length()
        if random_embedding:
            self.embedding = embeddings.get_initialized_embedding_layer(frozen=False)
            print(self.embedding)
        else:
            self.embedding = embeddings.get_initialized_embedding_layer(frozen=not fine_tune_embeddings)
            embedding_dim = embeddings.get_embedding_length()
        
        # Fully connected layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.hidden_layers.append(nn.Linear(embedding_dim, hidden_size))
            embedding_dim = hidden_size
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        self.dropoutword = dropoutword
        
        # Output layer for binary classification (2 classes: positive, negative)
        self.output_layer = nn.Linear(hidden_size, 2)
        
        # LogSoftmax for probability output
        self.log_softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
        - x: A batch of sentences (tensor of word indices).
        
        Returns:
        - log probabilities for each class (positive/negative).
        """
        # Embed the word indices
        embedded = self.embedding(x)

        # Apply word dropout: randomly zero out some word embeddings
        if self.training:  # Only apply dropout during training
            mask = (torch.rand(embedded.shape[:2]) > self.dropoutword).float().unsqueeze(2).to(embedded.device)
            embedded = embedded * mask  # Zero out some word embeddings
        
        # Average the embeddings (average along the sentence length dimension)
        sentence_embedding = embedded.mean(dim=1)
        
        # Pass through hidden layers with ReLU activations and dropout
        for layer in self.hidden_layers:
            sentence_embedding = F.relu(layer(sentence_embedding))
            sentence_embedding = self.dropout(sentence_embedding)
        
        # Pass through output layer and apply log softmax
        output = self.output_layer(sentence_embedding)
        return self.log_softmax(output)
    

class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, word_indexer, max_length=None):
        """
        Dataset class for the DAN model.
        
        Args:
        - infile: Path to the data file (train/dev/test).
        - word_indexer: An Indexer object to convert words to indices.
        - max_length: Optional, max length to pad or truncate sentences.
        """
        # Read the sentiment examples from the input file
        self.examples = read_sentiment_examples(infile)

        # Extract sentences and labels from the examples
        self.sentences = [ex.words for ex in self.examples]
        self.labels = [ex.label for ex in self.examples]

        self.word_indexer = word_indexer
        self.max_length = max_length

        # Convert sentences to indices
        self.indexed_sentences = self._index_sentences()

        # Convert labels to PyTorch tensors
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def _index_sentences(self):
        """
        Convert each sentence to a list of word indices.
        """
        indexed_sentences = []
        for sentence in self.sentences:
            # Convert words to indices, if the word is not found, use the UNK index
            indexed = [self.word_indexer.index_of(word) if self.word_indexer.index_of(word) != -1
                       else self.word_indexer.index_of("UNK") for word in sentence]
            
            # Optionally pad or truncate to max_length
            if self.max_length:
                indexed = indexed[:self.max_length]  # Truncate if too long
                indexed += [self.word_indexer.index_of("PAD")] * (self.max_length - len(indexed))  # Pad if too short
            
            indexed_sentences.append(indexed)
        
        # Convert to PyTorch tensor
        return [torch.tensor(indices, dtype=torch.long) for indices in indexed_sentences]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        """
        Return a tuple of (indexed sentence, label).
        """
        return self.indexed_sentences[idx], self.labels[idx]

 
class SentimentDatasetBPE(Dataset):
    def __init__(self, infile, bpe_vocab, merges, max_length=None):
        # Read sentiment examples using byte-level tokenization
        self.examples = read_sentiment_examples_bpe(infile)

        # Extract byte-level sentences and labels
        self.sentences = [ex.words for ex in self.examples]
        self.labels = [ex.label for ex in self.examples]
        self.bpe_vocab = bpe_vocab
        self.max_length = max_length
        self.pad_token_id = len(bpe_vocab)-1
        self.merges = merges

        # Tokenize and encode sentences using BPE
        self.indexed_sentences = self._index_sentences()

        # Convert labels to PyTorch tensors
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def _index_sentences(self):
        indexed_sentences = []
        for byte_sequence in self.sentences:
            # Use BPE encoding to convert the byte sequence to BPE tokens
            bpe_tokens = encode_bpe(byte_sequence, self.merges)

            # Optionally pad or truncate to max_length
            if self.max_length:
                bpe_tokens = bpe_tokens[:self.max_length]
                bpe_tokens += [self.pad_token_id] * (self.max_length - len(bpe_tokens))

            indexed_sentences.append(bpe_tokens)

        return [torch.tensor(indices, dtype=torch.long) for indices in indexed_sentences]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.indexed_sentences[idx], self.labels[idx]
    
    