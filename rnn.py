'''
This file will contain the rnn class and its methods
- initalize_parameters
- forward_pass
- backward_pass
- calculate_loss
- optimize

'''

class rnn:
    def __int__(self, X, y, num_layers, num_neurons):
        self.X = X
        self.y = y
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        