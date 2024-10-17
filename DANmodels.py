import torch
from torch import nn
import torch.nn.functional as F

class DAN(nn.Module):
    def __init__(self, embeddings, hidden_size=300, dropout=0.5, num_layers=2, fine_tune_embeddings=True):
        """
        Initialize the Deep Averaging Network (DAN).
        
        :param embeddings: Pre-trained embeddings (WordEmbeddings object from sentiment_data.py)
        :param hidden_size: The size of the hidden layer(s) in the network
        :param dropout: The dropout rate applied after the hidden layer
        :param num_layers: Number of hidden layers
        :param fine_tune_embeddings: Whether to fine-tune the embeddings during training
        """
        super(DAN, self).__init__()
        
        # Initialize the embedding layer with pre-trained embeddings
        vocab_size, embedding_dim = embeddings.shape
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=not fine_tune_embeddings)
        
        # Define layers for the feedforward neural network
        layers = []
        input_dim = embedding_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = hidden_size
        
        # Final classification layer
        layers.append(nn.Linear(hidden_size, 2))  # Output layer (2 classes for binary classification)
        layers.append(nn.LogSoftmax(dim=1))
        
        # Create the sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Look up embeddings for the input word indices
        embedded = self.embedding(x)
        
        # Average the word embeddings across the sentence (dim=1 is the sentence dimension)
        sentence_embedding = embedded.mean(dim=1)
        
        # Pass through the feedforward network
        return self.model(sentence_embedding)