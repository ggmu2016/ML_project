'''
This file contains the model class for building the rnn model, the functions will include:
    - Train
    - Test
'''

import sys
from preprocess import PreProcess
from rnn import RNN

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

    file_path = sys.argv[1]
    #train_split_size = sys.argv[2]
    #batch_size = sys.argv[3]
    #seq_length = sys.argv[4]

    p = PreProcess(file_path, 0.8, 32, 10)
    p.preprocess_data()


if __name__ == "__main__":
    main()