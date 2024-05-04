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
        self.num_outputs = np.shape(self.y)[-1]
        self.num_batches = num_batches
        self.batch_size = len(X) // num_batches
        self.learning_rate = learning_rate
        self.W_xh = None
        self.W_hh = None
        self.W_hy = None
        self.b_hh = None
        self.b_hy = None
        self.hidden_states = None
        self.y_model = []
        self.__initialize_parameters(seed=50)

    def __initialize_parameters(self, seed: int):
        np.random.seed(seed)
        self.W_xh = np.random.rand(self.num_features, self.num_neurons)
        self.W_hh = np.random.rand(self.num_neurons, self.num_neurons)
        self.W_hy = np.random.rand(self.num_neurons)
        self.b_hh = np.random.rand(self.num_neurons)
        self.b_hy = np.random.rand(np.shape(self.y)[-1])

    def sigmoid(self, x):
        f = 1 / (1 + np.exp(-1 * x))
        return f

    def d_sigmoid(self, x):
        f = 1 / (1 + np.exp(-1 * x))
        df = f * (1 - f)
        return df

    def forward_prop(self, X_seq):
        outputs = []
        self.hidden_states = []
        h_t = np.zeros((self.num_neurons,))

        for t in range(self.seq_length):
            a_t = self.b_hh + np.dot(X_seq[t], self.W_xh) + np.dot(h_t, self.W_hh)
            h_t = np.tanh(a_t)
            self.hidden_states.append(h_t)
            o_t = np.dot(h_t, self.W_hy) + self.b_hy
            outputs.append(o_t)
        # Pass through dense layer (using just 1 neuron)
        y_out = self.sigmoid(outputs[-1])
        self.hidden_states = np.array(self.hidden_states)  # shape: (num_neurons, num_features)
        return y_out

    def back_prop(self, X_seq, y_out, dL):
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        db_hh = np.zeros_like(self.b_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_hy = 0

        # Last hidden node only
        do_t = dL * self.d_sigmoid(y_out)
        dW_hy = np.array(self.hidden_states[-1])*do_t # same as dW_hy
        db_hy = do_t
        dh_t = do_t*self.W_hy
        dh_next = dh_t
        for t in range(self.seq_length - 1, -1, -1):
            h_t = self.hidden_states[t]
            da_t = (1 - (h_t * h_t))*dh_t
            db_hh += da_t
            dW_xh += np.dot(da_t, X_seq[t].T)
            if t != 0:
                dW_hh += np.dot(da_t, self.hidden_states[t - 1].T)
                dh_t = np.dot(self.W_hh.T, da_t)
                if t == self.seq_length - 1:
                    dh_t += dh_next
        self.update_weights(dW_xh, dW_hh, dW_hy, db_hh, db_hy)

    def calc_loss(self, y_target, y_model, loss_function="MSE"):
        if loss_function == "MSE":
            L = 0.5 * (y_target - y_model) ** 2
            dL = y_model - y_target
            return L, dL

    def update_weights(self, dW_xh, dW_hh, dW_hy, db_hh, db_hy):
        # Update weights and biases using SGD
        self.W_xh -= self.learning_rate * dW_xh
        self.W_hh -= self.learning_rate * dW_hh
        self.W_hy -= self.learning_rate * dW_hy
        self.b_hh -= self.learning_rate * db_hh
        self.b_hy -= self.learning_rate * db_hy

    def rnn(self):
        for batch in range(self.num_batches):
            y_model_seq=[]
            for seq in range(self.batch_size):
                X_seq = self.X[batch][seq]
                y_seq = self.y[batch][seq][0]
                y_out = self.forward_prop(X_seq)
                L, dL = self.calc_loss(y_target=y_seq, y_model=y_out)
                self.back_prop(X_seq, y_out, dL)
                y_model_seq.append(y_out)
                print('y_seq: ', y_seq)
                print('y_out: ', y_out)
            self.y_model.append(y_model_seq)

