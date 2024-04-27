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
    num_neurons = 6
    seq_length = 10
    num_features = 6
    num_batches = 264
    learning_rate = 0.001
    train_split_size = 0.8


    #file_path = sys.argv[1]
    file_path = './AAPL.csv'
    #train_split_size = sys.argv[2]
    #batch_size = sys.argv[3]
    #seq_length = sys.argv[4]

    p = PreProcess(file_path, train_split_size=0.8, batch_size=32, seq_length=10)
    p.preprocess_data()
    X = p.X_train_batches
    y = p.y_train_batches
    #print(np.array(p.X_train_batches).shape)
    #print(np.array(p.y_train_batches).shape)
    model = RNN(X=X, y=y, num_layers=num_layers, num_neurons=num_neurons, seq_length=seq_length,
                num_features=num_features, num_batches=num_batches, learning_rate=learning_rate)
    model.rnn()
    print(np.array(model.y_model).shape)
    print(np.shape(y))


if __name__ == "__main__":
    main()