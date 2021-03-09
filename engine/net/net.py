'''
network classes and operations
Dung Tran: 10/30/2020 Update: 
'''


import numpy as np

    
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
        Y1 = self.W*inputs + self.b
        

        
    def flatten(self):
        'flatten a layer into a sequence of operations'
        
        ops = [AM(self.W, self.b)]
        for i in range(self.num_neurons):
            if self.f == 'ReLU' or self.f == 'PosLin':
                ops.append(stepReLU(i))           
        return ops   

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

    def sample(self, inputs):
        'sampling the network with multiple input vectors'

        pass
        
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
        return ops
        
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
            W = np.random.rand(arch[i], arch[i-1])
            b = np.random.rand(arch[i], 1)
            if funs is None:
                L = Layer(W, b, 'ReLU')
            else:
                L = Layer(W, b, funs[i-1])
            layers.append(L)
        return FNN(layers)

class Test(object):
    'tests for this net module'
    
    def test_AM():
        'test affine mapping operation'
        # create a StarNode that has two children 
        # Root Node: [x; y] = [0; 0] + [1;0]*x + [0; 1]*y
        # LelfChild: [x; y], x >= 0: 
        # RightChild: [0; y], x <= 0 
    
        Eq = np.array([[0, 1, 0], [0, 0, 1]])
        Ineq = None
    
        root = StarNode(Eq, Ineq)
    
        W = np.array([[1, 2], [-1, 1]])
        b = np.array([[-1],[0]])
        op1 = AM(W, b)
    
        s1, s2 = op1.execute(root)
        s1.print()
        s2.print()
        
    def test_stepReLU():
        'test stepReLU operation'
        # create a StarNode that has two children 
        # Root Node: [x; y] = [0; 0] + [1;0]*x + [0; 1]*y
        # LelfChild: [x; y], x >= 0: 
        # RightChild: [0; y], x <= 0 
    
        Eq = np.array([[0, 1, 0], [0, 0, 1]])
        Ineq = None
    
        root = Star(Eq, Ineq)
        op1 = stepReLU(0)
    
        rs = op1.execute_single_input(root)
        print('rs has {} stars'.format(len(rs)))
        rs[0].print()
        rs[1].print()
        
        op2 = stepReLU(1)
        rs2 = op2.execute(rs)
        print('rs2 has {} stars'.format(len(rs2)))
        rs2[0].print()
        rs2[1].print()
        rs2[2].print()
        rs2[3].print()
        
        
    def test_Layer():
        'test layer object'
        
        W = np.array([[1, 2], [0.5, 1]])
        b = np.array([[1],[-1]])
        
        L = Layer(W, b, 'ReLU')
        ops = L.flatten()
        print('Number of operations: ', len(ops))
        print(ops)
        
    def test_FNN():
        'test FNN object'
        
        W1 = np.array([[1, 2], [0.5, 1]])
        b1 = np.array([[1],[-1]])
        L1 = Layer(W1, b1, 'ReLU')
        
        W2 = np.array([[0.5, 1], [1.5, 2]])
        b2 = np.array([[1],[0]])
        L2 = Layer(W2, b2, 'PosLin')
        
        layers = [L1, L2]
        net = FNN(layers)
        net.print()
        ops = net.flatten()
        print('Flattened network has {} operations'.format(len(ops)))
        print(ops)
        print(net)
        
        tree = net.symbolic_execution()
        print(tree)
        leaf_nodes = tree.get_leaf_nodes()
        print('Number of leaf nodes: ', len(leaf_nodes))
        
        leaf_nodes[1].print()
        Ineqs = leaf_nodes[1].get_all_Ineqs()
        if leaf_nodes[1].has_parent():
            print('Has parent')
            
    def test_FNN_rand():
        'test random function'
        
        arch = [2, 2]
        funs = None
        net = FNN.rand(arch, funs)
        net.print()
        ops = net.flatten() # network is equivalent to a set of operations
        Eq = np.identity(2)
        Eq = np.concatenate((Eq, np.zeros((2,1))), axis=1)
        Ineq = None
        S = Star(Eq, Ineq)
        print('Number of operations: ', len(ops))
        print(ops[0])
        print(ops[1])
        print(ops[2])
        # symbolic input set
        print('symbolic input set S:')
        S.print()
        # propagate this symbolic input set through all operations (network)
        # print('after affine mapping operation 0:')
        # S1 = ops[0].execute([S])
        # S1[0].print()
        # print('after stepReLU operation 1')
        # S2 = ops[1].execute([S1[0]])
        # S2[0].print()
        # # print('after stepReLU operation 2')
        # S3 = ops[2].execute(S2)
        # S3[0].print()
        # # property need to be infered: y1 > 0
        # Eq = S3[0].get_Eq()
        # # Eq = x + y + c > 0 <=> -x - y < c
        # new_Ineq = -np.array([Eq[0, :]])
        # new_Ineq[0, 2] = - new_Ineq[0, 2]

        # final constraint on symbolic input set
        # print('Final constraints on inputs such that the property is satisfied:')
        # new_Ineqs = np.concatenate((new_Ineq, S3[0].get_Ineqs()))
        # print(new_Ineqs)
        
        # S4 = ops[3].execute(S3)
        # print(S4)
        #S4[0].print()
        #S5 = ops[4].execute(S4)
        #print(S5)
        #S6 = ops[5].execute
        
        #sym_tree = net.execute()
        #print(sym_tree)


if __name__ == '__main__':
    
    #Test.test_stepReLU()
    #Test.test_AM()
    #Test.test_Layer()
    #Test.test_FNN()
    Test.test_FNN_rand()   
        
          
        
        
        
        
        
        
            
        
