'''
This file will contain the rnn class and its methods
- initalize_parameters
- forward_pass
- backward_pass
- calculate_loss
- optimize

'''
import numpy as np



class rnn:
    def __int__(self, X, y, num_layers, num_neurons, seq_length):
        self.X = X
        self.y = y
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.seq_length = seq_length
    def initialize_parameters(self, W_xh, W_hh, W_hy, b_hh):
        return 0
    def feedforward(self, W_xh, W_hh, W_hy,b_hh):
        return 0
    def back_prop(self,W_xh, W_hh, W_hy, l, dl, h_t, h_raw):
        for t in range(self.seq_length):
            dl_dhnext = 1 # temp
            dl_yt = dl
            dl_dWhy = np.matmul(dl_yt, np.array(h_t).transpose())
            dl_ht = np.matmul(np.array(W_hy).transpose(),dl_yt) + dl_dhnext
            dl_hraw = (1-np.matmul(np.arrar(h_t),np.array(h_t))) * dl_ht
            dl_dbh = dl_hraw
            dl_dWxh = dl_hraw * self.X[t] # temp: needs to be dl_hraw * self.X[t]^T
            dl_dWhh = dl_hraw * h_t[t-1]  # temp: needs to be dl_hraw * h[t-1]^T
            dl_dhnext = np.array(W_hh).transpose() * dl_hraw # temp: needs to be W_hh^T * dl_hraw

    def update_weights(self):
        return 0

