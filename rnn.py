'''
This file will contain the rnn class and its methods
- initalize_parameters
- forward_pass
- backward_pass
- calculate_loss
- optimize
'''

import numpy as np

class RNN:
    def __int__(self, X, y, num_layers, num_neurons, seq_length, num_features):
        self.X = X
        self.y = y
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.seq_length = seq_length
        self.num_features = num_features
    def initialize_parameters(self, W_xh, W_hh, W_hy, b_hh):
        Wxh = np.random.rand(self.num_features, self.num_neurons)
        W_hh = np.random.rand(self.num_neurons,self.num_neurons)
        W_hy = np.random.rand(self.num_neurons, self.num_features)
        b_hh = np.random.rand(self.num_neurons)
    def feedforward(self, W_xh, W_hh, W_hy,b_hh):
        return 0
    def back_prop(self, W_xh, W_hh, W_hy, dL, h_t):
        for t in range(self.seq_length-1,-1,-1):
            if t == (self.seq_length-1):
                dh_next = 0  # temp
            dW_hy = np.matmul(np.array(dL), np.array(h_t[t]).transpose())
            dh_t = np.matmul(np.array(W_hy).transpose(), np.array(dL)) + np.array(dh_next)
            dh_raw = np.matmul((1-np.matmul(np.arrar(h_t[t]), np.array(h_t[t]))), dh_t)
            db_hh = dh_raw
            dW_xh = np.matmul(dh_raw, np.array(self.X[t]).transpose())
            dW_hh = np.matmul(dh_raw, np.array(h_t[t-1]).transpose())
            dh_next = np.matmul(np.array(W_hh).transpose(), dh_raw)
    def calc_loss(self, y_target, y_model, loss_function:str):
        if loss_function == "MSE":
            L = 0.5*(y_target - y_model)**2
            dL = y_target - y_model
            return L, dL


    def update_weights(self):
        return 0

