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

    def __init__(self, X, y, num_layers, num_neurons, seq_length, num_features, num_batches, learning_rate):
        self.X = X
        self.y = y
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.seq_length = seq_length
        self.num_features = num_features
        self.num_batches = num_batches
        self.batch_size = len(X)/num_batches
        self.learning_rate = learning_rate
        self.W_xh = np.array([])
        self.W_hh = np.array([])
        self.W_hy = np.array([])
        self.b_hh = np.array([])
        self.__initialize_parameters(seed=50)

    def __initialize_parameters(self, seed: int):
        np.random.seed(seed)
        self.W_xh = np.random.rand(self.num_features, self.num_neurons)
        self.W_hh = np.random.rand(self.num_neurons,self.num_neurons)
        self.W_hy = np.random.rand(self.num_neurons, self.num_features)
        self.b_hh = np.random.rand(self.num_neurons)
    def forward_prop(self, W_xh, W_hh, W_hy, b_hh):
        outputs = []
        hidden_states = []
        h_t = np.zeros((self.num_neurons,))

        for t in range(self.seq_length):
            a_t = b_hh + np.dot(self.X[t], W_xh) + np.dot(h_t, W_hh)
            h_t = np.tanh(a_t)
            hidden_states.append(h_t)
            output_t = np.dot(h_t, W_hy)
            outputs.append(output_t)

        return np.array(outputs), np.array(hidden_states)

    def back_prop(self, dL, h_t):
        for t in range(self.seq_length-1,-1,-1):
            if t == (self.seq_length-1):
                dh_next = 0  # temp
            dW_hy = np.matmul(dL, h_t[t].T)
            dh_t = np.matmul(self.W_hy.T, dL) + dh_next
            dh_raw = np.matmul((1-np.matmul(np.arrar(h_t[t]), np.array(h_t[t]))), dh_t)
            db_hh = dh_raw
            dW_xh = np.matmul(dh_raw, np.array(self.X[t]).transpose())
            dW_hh = np.matmul(dh_raw, np.array(h_t[t-1]).transpose())
            dh_next = np.matmul(np.array(self.W_hh).transpose(), dh_raw)

        return dW_xh, dW_hh, dW_hy, db_hh
    def calc_loss(self, y_target, y_model, loss_function = "MSE"):
        if loss_function == "MSE":
            L = 0.5*(y_target - y_model)**2
            dL = y_target - y_model
            return L, dL
    def update_weights(self, dW_xh, dW_hh, dW_hy, db_hh):
        # Update weights and biases using SGD
        self.W_xh -= self.learning_rate * dW_xh
        self.W_hh -= self.learning_rate * dW_hh
        self.W_hy -= self.learning_rate * dW_hy
        self.b_hh -= self.learning_rate * db_hh
    def rnn(self, W_xh, W_hh, W_hy, b_hh):
        output, h_t = self.forward_prop(W_xh, W_hh, W_hy,b_hh)
        L, dL = self.calc_loss(y_target=self.y[seq], y_model=output)
        dW_xh, dW_hh, dW_hy, db_hh = self.back_prop(W_xh, W_hh, W_hy, dL, h_t)
        self.update_weights(dW_xh, dW_hh, dW_hy, db_hh)

        return output


