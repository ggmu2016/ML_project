'''
This file contains the model class for building the rnn model, the functions will include:
    - Train
    - Test
'''

import sys
from preprocess import PreProcess
from rnn import RNN
import numpy as np

class Model:
    def __init__(self, epochs: int, learning_rate: float, loss_function: str):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss_function = loss_function
    def train(self):
        pass

    def test(self):
        pass


def main():

    """
    # get commandline args
    if len(sys.argv) != :
        print("Usage: python model.py <file_path> <train_split_size> <batch_size> <seq_length>....")
        sys.exit(1)
    """
    # Model Parameters
    num_layers = 2
    num_neurons = 5
    seq_length = 10
    num_features = 5
    num_batches = 264
    learning_rate = 0.3
    train_split_size = 0.8


    #file_path = sys.argv[1]
    #file_path = './AAPL.csv'
    file_url = 'https://raw.githubusercontent.com/daryaanbar/AAPL-Stock-Data/main/AAPL.csv'
    #train_split_size = sys.argv[2]
    #batch_size = sys.argv[3]
    #seq_length = sys.argv[4]

    p = PreProcess(file_url, train_split_size=0.8, batch_size=32, seq_length=10)
    p.preprocess_data()
    print(np.array(p.X_train_batches).shape)
    print(np.array(p.y_train_batches).shape)
    X_train = p.X_train_batches  # shape: (num_batches, batch_size, seq_length, num_features)
    y_train = p.y_train_batches  # shape: (num_batches, batch_size, 1)

    model = RNN(X=X_train, y=y_train, num_layers=num_layers, num_neurons=num_neurons, seq_length=seq_length,
                num_features=num_features, num_batches=num_batches, learning_rate=learning_rate)
    model.rnn()


if __name__ == "__main__":
    main()