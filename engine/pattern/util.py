'''
Some useful operations to build binarytree from data
Use to construct symbolic execution of ReLU networks
Dung Tran: 12/11/2020
'''

from binarytree import Node, get_parent, build
import numpy as np

def build_pattern_tree(samples):
    'build on-off pattern of a neuron network based on samples data'

    isinstance(samples, np.ndarray), 'error: samples is not an ndarray'
    min_samples = np.unique(samples, axis=0) # unique samples (minimized set of samples)
    n, m = min_samples.shape;   # n is the number of samples, m is the number of neurons
    root = Node(2)    
    patterns = min_samples.tolist()
    for i in range(n):
        root = add_pattern(root, patterns[i])
    cover_rate = min_samples.shape[0]/2**m  # percentage of coverage of the worst-case on-off pattern tree
    return root, cover_rate, min_samples
    

def add_pattern(root, pat):
    'add a new pattern to a pattern tree'

    isinstance(pat, list), 'error: new pattern is not an list'
    isinstance(root, Node), 'error: pattern tree is not a Node type'

    if pat and pat[0] == 0:
        pat.pop(0)
        if not root.left:
            root.left = Node(0)
        root.left = add_pattern(root.left, pat)

    if pat and pat[0] == 1:
        pat.pop(0)
        if not root.right:
            root.right = Node(1)
        root.right = add_pattern(root.right, pat)
    
    return root

def test_build_pattern_tree():
    'test get_on_off_pattern function'

    n_neurons = 4
    n_samples = 50

    samples = np.random.randint(2, size=(n_samples,n_neurons)) # random on-off pattern
    T, cover_rate, min_samples = build_pattern_tree(samples) # on-off pattern
    print(T)
    print('Cover rate = {}'.format(cover_rate))
    print(min_samples)

def test_add_pattern():
    'test add_pattern function'

    root = Node(2)
    pat = [0, 1, 0, 1]

    root1 = add_pattern(root, pat)
    print(root1)
    pat2 = [0, 0, 1, 0]
    root2 = add_pattern(root1, pat2)
    print(root2)
    pat3 = [0, 0, 1, 1]
    root3 = add_pattern(root2, pat3)
    print(root3)
    
if __name__ == '__main__':

    test_build_pattern_tree();
    #test_add_pattern();
