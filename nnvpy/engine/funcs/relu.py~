'''
Methods for ReLU activation functions
Dung Tran: 12/12/2020
'''

import numpy as np

class ReLU(object):
    'A ReLU activation function object'

    @staticmethod
    def eval(inputs):
        'apply ReLU activation function on an input array'

        assert isinstance(inputs, np.ndarray), 'input is not an ndarray'
        output = np.where(inputs > 0, inputs, 0)
        return output

    @staticmethod
    def pattern(inputs):
        'return the on-off patterns when applying ReLU activation function on an input array'

        assert isinstance(inputs, np.ndarray), 'input is not an ndarray'
        pattern = np.where(inputs > 0, 1, 0)
        return pattern

        
