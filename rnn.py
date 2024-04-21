'''
This file will contain the rnn class and its methods
- initalize_parameters
- forward_pass
- backward_pass
- calculate_loss
'''

import numpy as np

class RNN:
    def __int__(self, X, y, num_neurons, seq_length, num_features, num_batches, learning_rate):
        '''
        :param X: Input; shape: (num features, sequence_length)
        :param y: Target Output; shape: (1)
        :param num_neurons: Number of neurons connected to each RNN hidden layer node
        :param seq_length: length of chosen time-series data (sequence length, eg: 20 days stock market data)
        :param num_features:
        :param num_batches:
        :param learning_rate:
        :return:
        '''
        self.X = X
        self.y = y
        self.num_neurons = num_neurons
        self.seq_length = seq_length
        self.num_features = num_features
        self.num_batches = num_batches
        self.batch_size = len(X)/num_batches
        self.learning_rate = learning_rate
    def initialize_parameters(self):
        Wxh = np.random.rand(self.num_features, self.num_neurons)
        W_hh = np.random.rand(self.num_neurons,self.num_neurons)
        W_hy = np.random.rand(self.num_neurons, self.num_features)
        b_hh = np.random.rand(self.num_neurons)
    def feedforward(self, W_xh, W_hh, W_hy,b_hh):
        output, h_t = 1, 1
        return output, h_t
    def back_prop(self, W_xh, W_hh, W_hy, b_hh, h_t, y_model):
        L, dL = self.calc_loss(y_model)
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

        # Update Weights (gd_momentum)
        V_Whx, W_xh = self.gd_momentum(dW_xh, W_xh, momentum=0.3, velocity=V_Whx)
        V_Whh, W_hh = self.gd_momentum(dW_hh, W_hh, momentum=0.3, velocity=V_Whh)
        V_Why, W_hy = self.gd_momentum(dW_hy, W_hy, momentum=0.3, velocity=V_Why)
        V_bhh, b_hh = self.gd_momentum(db_hh, b_hh, momentum=0.3, velocity=V_bhh)

        return W_xh, W_hh, W_hy, b_hh
    def calc_loss(self, y_model, loss_function = "MSE"):
        if loss_function == "MSE":
            L = 0.5*(self.y - y_model)**2
            dL = self.y - y_model
            return L, dL
    def update_weights(self, dW_hx, dW_hh, dW_hy, db_hh):
        pass

    def gd_momentum(self, delta, old_value, momentum=0.3, velocity=0):
        velocity_new = velocity*momentum + self.learning_rate*delta
        new_value = old_value - velocity_new
        return velocity_new, new_value

    def rnn(self, W_xh, W_hh, W_hy, b_hh):
        output, h_t = self.feedforward(W_xh, W_hh, W_hy, b_hh)
        W_xh, W_hh, W_hy, b_hh = self.back_prop(W_xh, W_hh, W_hy, h_t, output)

        return output


