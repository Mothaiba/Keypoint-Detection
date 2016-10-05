import cv2
import numpy as np

class NeuralNet:

    SIGMOID, TANH = 0, 1
    activation_map = [staticmethod.sigmoid_activation, staticmethod.tanh_activation]


    '''For now, I assume all the hidden layers have the same amount of neurons (n_hidden)'''
    def __init__(self, n_hidden_layers=1, n_input=2, n_output=2,
                 n_hidden=2, activition_function = SIGMOID):
        self.n_hidden_layers = n_hidden_layers
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden

        '''learning rate'''
        self.alpha = 0.5

        '''First weight-set connects input layer to hidden layer 1'''
        self.i_weight = np.random.random_sample((n_input, n_hidden)) * 2. - 1
        # self.i_weight = np.array([[.15, .25], [.2, .3]])
        # print 'i_weight:', self.i_weight

        '''Other weight-sets connect 2 consecutive hidden layers together'''
        '''Note: h_weight[i, j, k] mean the weight from layer i-th to layer (i + 1)-th'''
        self.h_weight = np.random.random_sample((n_hidden_layers - 1, n_hidden, n_hidden)) * 2. - 1
        # print 'h_weight:', self.h_weight

        '''Last weight-set connects the last hidden layer to output layer'''
        self.o_weight = np.random.random_sample((n_hidden, n_output)) * 2. - 1
        # self.o_weight = np.array([[.4, .5], [.45, .55]])
        # print 'o_weight:', self.o_weight

        '''Biases is attached with hidden layers and output layer'''
        self.h_bias = np.random.random_sample((self.n_hidden_layers, self.n_hidden)) * 2. - 1
        self.o_bias = np.random.random_sample(self.n_output) * 2. - 1
        # self.h_bias = np.array([[.35, .35]])
        # self.o_bias = np.array([.6, .6])
        # print 'h_bias:', self.h_bias
        # print 'o_bias:', self.o_bias

        self.x = None
        self.h_net = np.zeros((self.n_hidden_layers, self.n_hidden), dtype=float)
        self.h_out = np.zeros((self.n_hidden_layers, self.n_hidden), dtype=float)
        self.y_net = np.zeros(self.n_output, dtype=float)
        self.y_out = np.zeros(self.n_output, dtype=float)

    def feedForward(self, x):
        # print 'feedForward'

        assert len(x) == self.n_input, ">>ERROR<< len(x) is different from self.n_input"

        self.x = np.array(x)

        '''Feed from input layer to the first hidden layer'''
        for idx in range(self.n_hidden):
            self.h_net[0, idx] = np.dot(self.x, self.i_weight[:, idx]) + self.h_bias[0, idx]

        # print 'h_net:', self.h_net[0]

        self.h_out[0] = self.ReLU_activation(self.h_net[0])
        # print 'h_out:', self.h_out[0]

        '''Feed between 2 consecutive hidden layers'''
        for layer_idx in range(1, self.n_hidden_layers):
            for neuron_idx in range(self.n_hidden):
                self.h_net[layer_idx, neuron_idx] =\
                    np.dot(self.h_out[layer_idx - 1, :], self.h_weight[layer_idx - 1][:, neuron_idx])\
                    + self.h_bias[layer_idx, neuron_idx]

            self.h_out[layer_idx] = self.ReLU_activation(self.h_net[layer_idx])

        '''Feed from the last hidden layer to output layer'''
        for idx in range(self.n_output):
            self.y_net[idx] = np.dot(self.h_out[self.n_hidden_layers - 1, :], self.o_weight[:, idx]) \
                              + self.o_bias[idx]
        self.y_out = self.ReLU_activation(self.y_net)
        print 'y_out:', self.y_out

    def backPropagation(self, target):
        # print 'Back Propagation'
        assert len(target) == self.n_output, ">>ERROR<< len(y) is different from self.n_output"
        target = np.array(target)

        '''Evaluate error'''
        error = 0.5 * np.power(target - self.y_out, 2)
        # print 'error:', error
        print 'total error:', np.sum(error)

        '''Back-propagate from output layer, and evaluate error for the next phase (last hidden layer)'''
        new_error = np.zeros(self.n_hidden, dtype=float)
        for j in range(self.n_output):
            '''Calculate error for the last hidden layer'''
            new_error += (target[j] - self.y_out[j]) * self.derivative_ReLU(self.y_net[j])\
                         * self.o_weight[:, j]

            '''Optimize weights'''
            self.o_weight[:, j] += self.alpha * (target[j] - self.y_out[j])\
                                   * self.derivative_ReLU(self.y_net[j])\
                                   * self.h_out[self.n_hidden_layers - 1, :]

        '''Back-propagate and evaluate error for pairs the hidden layers'''
        for k in range(self.n_hidden_layers - 2, -1, -1):
            error, new_error = new_error, np.zeros(self.n_hidden, dtype=float)

            for j in range(self.n_hidden):
                new_error += error[j] * self.derivative_ReLU(self.h_net[k + 1, j])\
                             * self.h_weight[k, :, j]

                self.h_weight[k, :, j] += self.alpha * error[j]\
                                          * self.derivative_ReLU(self.h_net[k + 1, j])\
                                          * self.h_out[k, :]

        '''Back-propagate to the input layer'''
        error, new_error = new_error, np.zeros(self.n_hidden, dtype=float)

        for j in range(self.n_hidden):
            # self.i_weight[:, j] += self.alpha * self.i_weight[:, j] * error[j]
            self.i_weight[:, j] += self.alpha * error[j]\
                                   * self.derivative_ReLU(self.h_net[0, j])\
                                   * self.x[:]
        # print 'i_weight:', self.i_weight
        '''DONE'''


    '''This implementation use sigmoid function for Activation'''
    @staticmethod
    def sigmoid_activation(self, val):
        '''val needs to be (a scalar) or (a numpy array)'''
        # return 1. / (1. + np.exp(val * -1))

    @staticmethod
    def derivative_sigmoid(self, val):
        '''val needs to be a scalar'''
        sig = NeuralNet.sigmoid_activation(val)
        return sig * (1. - sig)

    '''Tanh activation function'''
    @staticmethod
    def tanh_activation(self, val):
        return np.tanh(val)

    @staticmethod
    def derivative_tanh(self, val):
        tanh = NeuralNet.tanh_activation(val)
        return 1. - tanh * tanh

    '''ReLU activation function'''
    def ReLU_activation(self, val):
        if np.isscalar(val):
            return val if val >= 0 else 0
        tmp = np.copy(val)
        tmp[tmp < 0] = 0
        return tmp

    def derivative_ReLU(self, val):
        return 1. if val >= 0 else 0


if __name__ == '__main__':
    nn = NeuralNet(n_hidden_layers=1, n_hidden=11)
    x = [0.05, 0.1]
    y = [.01, .99]

    for i in range(2):
        nn.feedForward(x)
        nn.backPropagation(y)