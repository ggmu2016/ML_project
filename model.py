import sys
from preprocess import PreProcess
from rnn import RNN
import numpy as np
from sklearn.metrics import mean_squared_error as MSE


class Model:
    def __init__(self, epochs: int, learning_rate: float, num_features: int,
                 num_neurons, batch_size: int, seq_length: int, train_test_split):
        self.seq_length = seq_length
        self.num_neurons = num_neurons
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_features = num_features
        self.learning_rate = learning_rate
        # preprocessing the data
        file_url = 'https://raw.githubusercontent.com/daryaanbar/AAPL-Stock-Data/main/AAPL.csv'
        p = PreProcess(file_url, train_split_size=train_test_split,
                        batch_size=self.batch_size, seq_length=self.seq_length)
        p.preprocess_data()
        # Initializing the rnn model
        self.X_train = p.X_train_batches  # shape: (num_batches, batch_size, seq_length, num_features)
        self.y_train = p.y_train_batches  # shape: (num_batches, batch_size, 1)
        self.model = RNN(X=self.X_train, y=self.y_train, num_neurons=self.num_neurons, seq_length=self.seq_length,
                         num_features=self.num_features, batch_size=self.batch_size, learning_rate=self.learning_rate)
        self.X_test = p.X_test
        self.y_test = p.y_test

    def train(self):
        self.model.train(self.epochs)
        y_model_train = self.model.y_model
        x1 = np.array(self.model.y).reshape(-1, 1)
        x2 = np.array(y_model_train).reshape(-1, 1)
        print('y_true_shape: ', np.shape(x1))
        print('y_pred_shape: ', np.shape(x2))
        train_mse = self.accuracy(y_true=x1, y_pred=x2)
        return train_mse

    def accuracy(self, y_true, y_pred):
        print(np.shape(y_true))
        print(np.shape(y_pred))
        mse = MSE(y_true, y_pred)
        return mse

    def test(self):
        y_model_test = self.model.test(self.X_test)



def main():
    """
    # get commandline args
    if len(sys.argv) != :
        print("Usage: python model.py <file_path> <train_split_size> <batch_size> <seq_length>....")
        sys.exit(1)
    """
    # Model Parameters
    train_test_split = 0.8
    seq_length = 10
    num_features = 5
    learning_rate = 0.3
    epochs = 20
    batch_size = 32
    num_neurons = 20

    # file_path = sys.argv[1]
    # file_path = './AAPL.csv'
    # train_split_size = sys.argv[2]
    # batch_size = sys.argv[3]
    # seq_length = sys.argv[4]

    rnn_model = Model(epochs=epochs, learning_rate=learning_rate, num_features=num_features, batch_size=batch_size,
                      seq_length=seq_length, train_test_split=train_test_split, num_neurons=num_neurons)

    train_accuracy = rnn_model.train()
    print(train_accuracy)


if __name__ == "__main__":
    main()
