# DANmodels.py

import torch
from torch import nn
import torch.nn.functional as F
from sentiment_data import read_sentiment_examples
from torch.utils.data import Dataset

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
