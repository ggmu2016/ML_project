Rohit Hejib (rnh220001), Shan Baig (sab180014), Darya Anbar (dxa200020)
CS 6375.001 Project


# Predicting Stock Price Using a Recurrent Neural Network 
This program implements a custom Recurrent Neural Network (RNN) for time series prediction.
The model is trained on Apple Inc. (AAPL) stock data to predict future adjusted closing prices.


## Requirements
This program has the following installation requirements:
  - python 3.x
  - numpy
  - pandas
  - requests
  - scikit-learn
  - matplotlib


## Usage
Before running, adjust the hyperparameters directly in the model.py file within the main() function.
The hyperparameters include:
  - training/test data split ratio        (train_test_split)
  - length of each sequence               (seq_length)
  - learning rate                         (learning_rate)
  - number of epochs                      (epochs)
  - number of sequences in each batch     (batch_size)
  - number of neurons                     (num_neurons)

To run the program: 'python model.py'
