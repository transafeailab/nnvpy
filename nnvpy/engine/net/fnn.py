'''
Feedforward neural network class and its methods
Dung Tran: 12/12/2020
'''

import numpy as np
from engine.layer.Layer import Layer
from engine.pattern.util import build_pattern_tree

class FNN(object):
    'feedforward neural network object'
    
    def __init__(self, layers):
        'feedforward neural network consist of a set of layers'
        
        assert isinstance(layers, list), 'error: layers should be a list'
        n = len(layers)
        # check if layer(i) is a Layer object
        N = 0
        for i in range(n):
            assert isinstance(layers[i], Layer), 'error: {} th object is not a Layer object'.format(i)
            N = N + layers[i].num_neurons
        
        self.layers = layers
        self.num_neurons = N
        self.num_layers = n
        self.num_inputs = layers[0].num_inputs
        self.num_outputs = layers[n-1].num_neurons
        self.ops = []

    def sample(self, inputs):
        'sampling the network with multiple input vectors'

        assert isinstance(inputs, np.ndarray), 'error: inputs is not an ndarray'
        assert inputs.shape[0] == self.num_inputs, 'error: the network and the input array have different number of inputs'
        Y = inputs;
        for i in range(self.num_layers):
            Y = self.layers[i].sample(Y)
        return Y

    def pattern(self, inputs):
        'get on-off pattern of the neurons corresponding to input vectors'

        assert isinstance(inputs, np.ndarray), 'error: inputs is not an ndarray'
        assert inputs.shape[0] == self.num_inputs, 'error: the network and the input array have different number of inputs'
        Y = inputs;
        pat = None;
        for i in range(self.num_layers):
            Y = self.layers[i].pattern(Y)
            if i==0:
                pat = Y
            else:
                pat = np.concatenate((pat, Y), axis=0)
        return pat

    def fuzz(self, n_samples, lb, ub, constr_mat=[], constr_vec=[]):
        'fuzz the network to get on-off pattern'
        
        assert isinstance(lb, np.ndarray), 'error: input lower bound is not an ndarray'
        assert isinstance(ub, np.ndarray), 'error: output upper bound is not an ndarray'
        assert lb.shape[1] == 1, 'error: input lower bound is not a vector'
        assert ub.shape[1] == 1, 'error: input upper bound is not a vector'
        assert lb.shape[0] == ub.shape[0], 'error: inconsistency between the input lower bound and upper bound'
        assert lb.shape[0] == self.num_inputs, 'error: inconsistency between the number of inputs of the lower/upper bound vectors and the network'
        assert n_samples >= 1, 'error: ivalid number of samples'
        
        has_constr = isinstance(constr_mat, np.ndarray) and isinstance(constr_vec, np.ndarray)
        if has_constr:
            assert constr_vec.shape[1] == 1, 'error: constraint vector is not a vector'
            assert constr_vec.shape[0] == constr_mat.shape[0], 'error: inconsistency between constraint matrix and vector'
            
            assert constr_mat.shape[1] == lb.shape[0], 'error: inconsistency between the number of inputs in the lower/upper bound vectors and the constraints'

        if has_constr: # fuzz the network with constrained inputs
            print('Fuzzing with constrained inputs has not been supported yet')
        else: # fuzs the network with lower bound and upper bound vector
            inputs = np.random.uniform(lb, ub, size=(self.num_inputs, n_samples))
            outputs = self.sample(inputs)
            pattern_samples = self.pattern(inputs)
            pattern_tree, coverage, min_paths = build_pattern_tree(np.transpose(pattern_samples))

        return outputs, pattern_tree, coverage, min_paths

            
    def print(self):
        'print this network info'
        
        print('============ Network Information =============')
        print('* Network type: feedforward')
        print('* Number of layers: ', self.num_layers)
        print('* Number of neurons: ', self.num_neurons)
        print('* Number of inputs: ', self.num_inputs)
        print('* Number of outputs: ', self.num_outputs)
        
    def flatten(self):
        'flatten this network'
        
        ops = []
        for i in range(self.num_layers):
            ops = ops + self.layers[i].flatten()
        self.ops = ops
        return ops

    def get_output_constr(self, pat):
        'get output set from a on-off pattern'

        if not self.ops:
            self.flatten()
        
        pass


        
    def execute(self):
        'symbolic execution of the network'
        
        c = np.zeros((self.num_inputs, 1))
        V = np.identity(self.num_inputs)
        Eq = np.concatenate((c, V), axis=1)
        Ineq = None
        root = Star(Eq, Ineq)   
        ops = self.flatten()
        n = len(ops)
        tree = [[root]]
        for i in range(n):
            if i == 0:
                input = [root]
            else:
                input = tree[i-1]
            rs = ops[i].execute(input)
            tree.append(rs)
        return tree
    
    @staticmethod   
    def rand(arch, funs):
        'randomly generate a ReLU network'
        
        assert isinstance(arch, list), 'error: architecture should be a list'
        assert len(arch) >= 2, 'error: network should have at least one layer'
        if isinstance(funs, list):
            if len(arch) != len(funs) + 2:
                raise Exception('Inconsistency between the number of layers and the number of activation functions')

        n = len(arch)
        layers = []
        for i in range(1, n):
            W = np.random.uniform(low=-1, high=1, size=(arch[i], arch[i-1]))
            b = np.random.uniform(low=-1, high=1, size=(arch[i], 1))
            if funs is None:
                L = Layer(W, b, 'ReLU')
            else:
                L = Layer(W, b, funs[i-1])
            layers.append(L)
        return FNN(layers)

    @staticmethod
    def test():
        'test for fnn class'                

        def test_rand():
            'test random function (randomly generate a FNN)'

            try:
                arch = [2, 2]
                funs = None
                net = FNN.rand(arch, funs)
                print('random method test: successful')
            except:
                print('random method test: failed')

        def test_sample():
            'test sample method'

            try:
                net = FNN.rand([2, 3, 3, 2], None)
                inputs = np.random.rand(2,5)
                Y = net.sample(inputs)
                print('sample method test: successfully')
            except:
                print('sample method test: failed')
                
        def test_pattern():
            'test pattern method'

            try:
                net = FNN.rand([2, 3, 3, 2], None)
                inputs = np.random.uniform(low=-5, high=1, size=(2,5))
                Y = net.pattern(inputs)
                print('pattern method test: successfully')
            except:
                print('pattern method test: failed')

        def test_fuzz():
            'test fuzz method'

            try:
                net = FNN.rand([2, 2, 2, 2], None)
                lb = np.array([[-3], [-3]]) # range [-3, 3]
                ub = np.array([[3], [3]])
                n_samples = 20
                outputs, pattern_tree, coverage, min_paths = net.fuzz(n_samples, lb, ub)
                print('============ On-Off Pattern Tree=============')
                print(pattern_tree)
                print('Coverage of the pattern tree: {}'.format(coverage))
                print('fuzz method test: successfully')
            except:
                print('fuzz method test: failed')
                
        def test_print():
            'test print method'

            try:
                net = FNN.rand([2, 2], None)
                net.print()
                print('print method test: successful')
            except:
                print('print method test: failed')

        def test_flatten():
            'test flatten method'
            
            try:
                net = FNN.rand([2, 3, 2], None)
                ops = net.flatten()
                print('flatten method test: successful')
            except:
                print('flattern method test: failed')

        # test all methods here
        methods = ['rand', 'sample', 'pattern', 'fuzz', 'print', 'flatten']  # name of methods need to be tested
        n = len(methods)
        print('++++++++++++ Test FNN Class +++++++++++++')
        for i in range(n):

            if methods[i] == 'rand':
                test_rand()
            elif methods[i] == 'sample':
                test_sample()
            elif methods[i] == 'pattern':
                test_pattern()
            elif methods[i] == 'fuzz':
                test_fuzz()
            elif methods[i] == 'print':
                test_print()
            elif methods[i] == 'flatten':
                test_flatten()
            else:
                raise ValueError("{}th method name is unknown".format(i))

