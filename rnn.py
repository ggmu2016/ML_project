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
        self.X = X  # shape: (num_batches, batch_size, seq_length, num_features)
        self.y = y  # shape: (num_batches, batch_size, 1)
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.seq_length = seq_length
        self.num_features = num_features
        self.num_batches = num_batches
        self.batch_size = len(X) // num_batches
        self.learning_rate = learning_rate
        self.W_xh = np.array([])
        self.W_hh = np.array([])
        self.W_hy = np.array([])
        self.b_hh = np.array([])
        self.hidden_states = None
        self.y_model = []
        self.__initialize_parameters(seed=50)

    def __initialize_parameters(self, seed: int):
        np.random.seed(seed)
        self.W_xh = np.random.rand(self.num_features, self.num_neurons)
        self.W_hh = np.random.rand(self.num_neurons, self.num_neurons)
        self.W_hy = np.random.rand(self.num_neurons, self.num_features)
        self.b_hh = np.random.rand(self.num_neurons)

    def forward_prop(self, X_seq):
        outputs = []
        output_t = -1
        self.hidden_states = []
        h_t = np.zeros((self.num_neurons,))

        for t in range(self.seq_length):
            a_t = self.b_hh + np.dot(X_seq[t], self.W_xh) + np.dot(h_t, self.W_hh)
            h_t = np.tanh(a_t)
            self.hidden_states.append(h_t)
            output_t = np.dot(h_t, self.W_hy)
            outputs.append(output_t)
        self.hidden_states = np.array(self.hidden_states)  # shape: (num_neurons, num_features)
        return output_t

    def back_prop(self, X_seq, dL):
        # Incoming X_seq is
        dW_xh = np.zeros(shape=(np.shape(X_seq)[1]))
        dW_hh = np.zeros(shape=(np.shape(self.hidden_states)[1]))
        dW_hy = np.array([])
        db_hh = np.array([])
        for t in range(self.seq_length - 1, -1, -1):
            if t == (self.seq_length - 1):
                dh_next = 0  # temp
            h_t = self.hidden_states[t]
            dW_hy += np.dot(dL, h_t.T)
            dh_t = np.dot(self.W_hy.T, dL) + dh_next
            dh_raw = (1 - (h_t * h_t))*dh_t
            print(dh_raw)
            db_hh += dh_raw
            dW_xh += np.dot(dh_raw, X_seq[t].T)
            dW_hh += np.dot(dh_raw, self.hidden_states[t - 1].T)
            dh_next = np.dot(self.W_hh.T, dh_raw)

        self.update_weights(dW_xh, dW_hh, dW_hy, db_hh)

    def calc_loss(self, y_target, y_model, loss_function="MSE"):
        if loss_function == "MSE":
            L = 0.5 * (y_target - y_model) ** 2
            dL = y_target - y_model
            return L, dL

    def update_weights(self, dW_xh, dW_hh, dW_hy, db_hh):
        # Update weights and biases using SGD
        self.W_xh -= self.learning_rate * dW_xh
        self.W_hh -= self.learning_rate * dW_hh
        self.W_hy -= self.learning_rate * dW_hy
        self.b_hh -= self.learning_rate * db_hh

    def rnn(self):
        for batch in range(self.num_batches):
            y_model_seq=[]
            for seq in range(self.batch_size):
                X_seq = self.X[batch][seq]
                print(X_seq)
                y_seq = self.y[batch][seq][0]
                print('y_seq: ', y_seq)
                output = self.forward_prop(X_seq)
                print('output: ', output)
                L, dL = self.calc_loss(y_target=y_seq, y_model=output)
                self.back_prop(X_seq, dL)
                y_model_seq.append(output)
                print('y_model_seq: ', y_model_seq)
            self.y_model.append(y_model_seq)

