'''
This file contains the model class for building the rnn model, the functions will include:
    - Train
    - Test
'''

class Model:
    def __init__(self, epochs: int, learning_rate: float, loss_function: str):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss_function = loss_function
    def train(self):
        pass

    def test(self):
        pass