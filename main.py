#main.py

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples, WordEmbeddings
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW
from DANmodels import DAN, SentimentDatasetDAN
from sentiment_data import read_word_embeddings


# Training function
def train_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        #X = X.float()
        if X.dtype != torch.long:
            X = X.long()  # Convert to LongTensor if it's not
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss


# Evaluation function
def eval_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(data_loader):
        #X = X.float()
        if X.dtype != torch.long:
            X = X.long()  # Convert to LongTensor if it's not
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss


# Experiment function to run training and evaluation for multiple epochs
def experiment(model, train_loader, test_loader, lr=0.0001, weight_decay=5e-5):
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    all_train_accuracy = []
    all_test_accuracy = []
    for epoch in range(100):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)

        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, optimizer)
        all_test_accuracy.append(test_accuracy)

        if epoch % 10 == 9:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}')
    
    return all_train_accuracy, all_test_accuracy


def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., BOW)')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden layer size')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for hidden layers')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--dropoutword', type=float, default=0.4, help='Dropout rate for word embeddings')
    parser.add_argument('--fine_tune_embeddings', action='store_true', help='Whether to fine-tune embeddings')
    parser.add_argument('--embedding_size', type=int, default=300, help='Embedding vector size')
    parser.add_argument('--max_length', type=int, default=100, help='Sentence length cap')
    parser.add_argument('--random_embedding', action='store_true', help='Whether to initialize random embeddings')
    # Training-related arguments
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load GloVe embeddings and get the word indexer
    if args.embedding_size == 300:
        glove_embeddings = read_word_embeddings("data/glove.6B.300d-relativized.txt")
    elif args.embedding_size == 50:
        glove_embeddings = read_word_embeddings("data/glove.6B.50d-relativized.txt")
    
    # Use the same word indexer for both GloVe and random embeddings
    word_indexer = glove_embeddings.word_indexer

    if args.random_embedding:
        # Initialize random embeddings with the same indexer
        print("Using random embeddings with embedding size:", args.embedding_size)
        rembedding = WordEmbeddings(word_indexer=word_indexer, random_init=True, embedding_dim=args.embedding_size)
    else:
        # Use GloVe embeddings
        rembedding = glove_embeddings  # Use pre-trained GloVe embeddings

    # Load pretrained GloVe embeddings
    #if args.embedding_size == 300:
    #    glove_embeddings = read_word_embeddings("data/glove.6B.300d-relativized.txt")
    #elif args.embedding_size == 50:
    #    glove_embeddings = read_word_embeddings("data/glove.6B.50d-relativized.txt")
    
    # Obtain the word indexer from the pretrained embeddings
    #word_indexer = glove_embeddings.word_indexer

    # Load dataset
    start_time = time.time()
    if args.model == "BOW":
        train_data = SentimentDatasetBOW("data/train.txt")
        dev_data = SentimentDatasetBOW("data/dev.txt")
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

    elif args.model == "DAN":
        max_length = args.max_length  # You can adjust this based on the dataset analysis
        train_data = SentimentDatasetDAN("data/train.txt", word_indexer, max_length=max_length)
        dev_data = SentimentDatasetDAN("data/dev.txt", word_indexer, max_length=max_length)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Data loaded in : {elapsed_time} seconds")

    # Print the parameters being used
    print(f"Training {args.model} model with the following parameters:")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Dropout (hidden layers): {args.dropout}")
    print(f"  Number of layers: {args.num_layers}")
    print(f"  Word dropout: {args.dropoutword}")
    print(f"  Fine-tune embeddings: {args.fine_tune_embeddings}")
    print(f"  Random embeddings: {args.random_embedding}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Max length: {args.max_length}")


    # Check if the model type is "BOW"
    if args.model == "BOW":
        # Train and evaluate NN2
        start_time = time.time()
        print('\n2 layers:')
        nn2_train_accuracy, nn2_test_accuracy = experiment(NN2BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Train and evaluate NN3
        print('\n3 layers:')
        nn3_train_accuracy, nn3_test_accuracy = experiment(NN3BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_train_accuracy, label='2 layers')
        plt.plot(nn3_train_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for 2, 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'train_accuracy.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_test_accuracy, label='2 layers')
        plt.plot(nn3_test_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for 2 and 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dev_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

        # plt.show()

    elif args.model == "DAN":
        # Load pre-trained GloVe embeddings (e.g., 50d)
        # Initialize the DAN model
        if args.random_embedding:
            rembedding = WordEmbeddings(word_indexer, random_init=True, embedding_dim=args.embedding_size)
            dan_model = DAN(embeddings=rembedding,
                            hidden_size=args.hidden_size,
                            dropout=args.dropout,
                            num_layers=args.num_layers,
                            dropoutword=args.dropoutword,
                            fine_tune_embeddings=args.fine_tune_embeddings,
                            random_embedding=args.random_embedding)
        else:
            dan_model = DAN(embeddings=glove_embeddings,
                            hidden_size=args.hidden_size,
                            dropout=args.dropout,
                            num_layers=args.num_layers,
                            dropoutword=args.dropoutword,
                            fine_tune_embeddings=args.fine_tune_embeddings)

        # Train and evaluate the DAN model
        print('\nTraining DAN model:')
        start_time = time.time()
        dan_train_accuracy, dan_test_accuracy = experiment(dan_model, train_loader, test_loader, lr=args.lr, weight_decay=args.weight_decay)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Trained in : {elapsed_time} seconds")

        # Plot results (similar to BOW)
        plt.figure(figsize=(8, 6))
        plt.plot(dan_train_accuracy, label='DAN Train Accuracy')
        plt.plot(dan_test_accuracy, label='DAN Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Dev Accuracy for DAN')
        plt.legend()
        plt.grid()
        plt.show()

        # Save the training accuracy figure
        train_dev_accuracy_file = 'train_dev_accuracy.png'
        plt.savefig(train_dev_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {train_dev_accuracy_file}")



if __name__ == "__main__":
    main()
