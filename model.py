import sys

from matplotlib import ticker

from preprocess import PreProcess
from rnn import RNN
import numpy as np
from sklearn.metrics import mean_squared_error as MSE
import matplotlib.pyplot as plt


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
        self.train_dates = p.dates_train.iloc[:, 0]
        self.test_dates = p.dates_test.iloc[:, 0]

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
        train_mse = MSE(y_true=x1, y_pred=x2)

        # Plot Train Data
        plt.plot(self.train_dates[:len(x1)], np.array(x1).flatten(), label='True')
        plt.plot(self.train_dates[:len(x1)], np.array(x2).flatten(), label='Model')
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(10))
        plt.xticks(rotation=45, fontsize=6)
        plt.legend()
        plt.grid(True)
        plt.title("AAPL Adjusted Close Prediction: Training Data")
        plt.xlabel('Date')
        plt.ylabel('Normalized Stock Price (Dollars)')
        plt.savefig('Train.png')
        plt.show()

        return train_mse

    def test(self):
        y_model_test = np.array(self.model.test(self.X_test)).flatten()

        # plot test data
        x_end = len(self.y_test)
        if len(self.y_test) % self.seq_length != 0:
            x_end = len(self.y_test) - (len(self.y_test) % self.seq_length)

        y_true = np.array(self.y_test).flatten()[0:x_end][0::self.seq_length]
        plt.plot(self.test_dates[0:x_end][0::self.seq_length], y_true, label='True')
        plt.plot(self.test_dates[0:x_end][0::self.seq_length], y_model_test, label='Model')
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(10))
        plt.xticks(rotation=45, fontsize=6)
        plt.title("AAPL Adjusted Close Prediction: Test Data")
        plt.xlabel('Date')
        plt.ylabel('Normalized Stock Price (Dollars)')
        plt.legend()
        plt.grid(True)
        plt.savefig('Test.png')
        plt.show()

        # calculate MSE for test data
        test_mse = MSE(y_true=y_true, y_pred=y_model_test)
        return test_mse


def main():

    #################### ENTER  ####################
    ################# MODEL PARAMETERS #####################
    train_test_split = 0.8
    seq_length = 10
    num_features = 4
    learning_rate = 0.05
    epochs = 15
    batch_size = 32
    num_neurons = 4
    #######################################################
    #######################################################

    rnn_model = Model(epochs=epochs, learning_rate=learning_rate, num_features=num_features, batch_size=batch_size,
                      seq_length=seq_length, train_test_split=train_test_split, num_neurons=num_neurons)

    train_mse = rnn_model.train()
    test_mse = rnn_model.test()

    print('Train MSE: ', train_mse)
    print('Test MSE: ', test_mse)


if __name__ == "__main__":
    main()
