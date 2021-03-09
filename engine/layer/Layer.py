'''
Layer class and its methods
Dung Tran: 12/12/2020
'''

import numpy as np
from engine.operation.operation import AM, stepReLU
from engine.funcs.relu import ReLU

class Layer(object):
    'a layer object for feedforward neural network'
    
    def __init__(self, W, b, f):
        
        # L: y = f(Wx +b), f is an activation function
        #    W: is weight matrix, b is bias vector
        
        assert isinstance(W, np.ndarray), 'error: weight matrix is not an ndarray'
        assert isinstance(b, np.ndarray), 'error: bias vector is not an ndarray'
        assert W.shape[0] == b.shape[0], 'error: inconsistency between weight matrix and bias vector'
        assert b.shape[1] == 1, 'error: bias vector has more than one column'
        
        self.W = W
        self.b = b
        self.num_neurons = W.shape[0] # number of neurons
        self.num_inputs = W.shape[1]  # number of inputs
        
        assert f == 'ReLU' or f == 'PosLin', 'error: unknown or unsupported activation function'
        self.f = f

    def sample(self, inputs):
        'sampling the layer with multiple inputs'

        assert isinstance(inputs, np.ndarray), 'error: inputs is not an ndarray'
        assert inputs.shape[0] == self.num_inputs, 'error: the layer and input array have different number of inputs'
        outputs = np.dot(self.W, inputs) + self.b
        if self.f == 'ReLU' or self.f == 'PosLin':
            outputs = ReLU.eval(outputs)
        else:
            raise ValueError('Unsupport activation function')
        return outputs

    def pattern(self, inputs):
        'get on-off pattern of all neurons in the layer corresponding to inputs'

        assert isinstance(inputs, np.ndarray), 'error: inputs is not an ndarray'
        assert inputs.shape[0] == self.num_inputs, 'error: the layer and input array have different number of inputs'
        outputs = np.dot(self.W, inputs) + self.b
        if self.f == 'ReLU' or self.f == 'PosLin':
            outputs = ReLU.pattern(outputs)
        else:
            raise ValueError('Unsupport activation function')
        return outputs
        
        
    def flatten(self):
        'flatten a layer into a sequence of operations'
        
        ops = [AM(self.W, self.b)]
        for i in range(self.num_neurons):
            if self.f == 'ReLU' or self.f == 'PosLin':
                ops.append(stepReLU(i))           
        return ops   
