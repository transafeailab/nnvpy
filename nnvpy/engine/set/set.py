'''
This contains different set representations and their methods
Dung Tran: 10/27/2020 Update: 
'''

import numpy as np
from scipy.optimize import linprog as lp

class StarNode (object):
    'A star node in a star tree'
    def __init__(self, Eq, Ineq):
    
        # A star node links to a parent node, and a child node and a silbling node
        # A star node has two parts:
        # 	1) Equality part: x = c + a1 * v1 + a2 * v2 + ...+ am * vm: Eq = [c v1 v2 ... vm]
        #   2) Inequality part: C * a <= d: Ineq = [C d]
		
        self.parent = None
        self.left_child = None
        self.right_child = None
        self.sibling = None
        self.ID = None
        self.loc = None
        self.Dim = None
		
        assert isinstance(Eq, np.ndarray), 'error: Equality of is not an ndarray'
		
        self.Eq = Eq
        self.Dim = Eq.shape[0]
        self.Ineq = Ineq
        
    def print(self):
        'print the information of this StarNode'
        
        print('========= StarNode Info ==========')
        print('*Node ID: ', self.get_ID())
        print('*Loc: ', self.get_loc())
        print('*Eq: ', self.get_Eq())
        print('*Ineq: ', self.get_Ineq())
        
        
        if self.parent is not None:
            print('*Parent:')
            print('    ID: ', self.parent.get_ID())
            print('    Loc: ', self.parent.get_loc())
            print('    Equality:', self.parent.get_Eq())
            print('    Inequality:', self.parent.get_Ineq())
        
        if self.sibling is not None:
            print('*Sibling:')
            print('    ID: ', self.sibling.get_ID())
            print('    Loc: ', self.sibling.get_loc())
            print('    Equality:', self.sibling.get_Eq())
            print('    Inequality:', self.sibling.get_Ineq())
        
        if self.left_child is not None:
            print('*Left child:')
            print('    ID: ', self.left_child.get_ID())
            print('    Loc: ', self.left_child.get_loc())
            print('    Equality:', self.left_child.get_Eq())
            print('    Inequality:', self.left_child.get_Ineq())
        
        if self.right_child is not None:
            print('*Right child:')
            print('    ID: ', self.right_child.get_ID())
            print('    Loc: ', self.right_child.get_loc())
            print('    Equality:', self.right_child.get_Eq())
            print('    Inequality:', self.right_child.get_Ineq())
        
        
    def set_parent(self, parent):
        'set a parent node'
        assert isinstance(parent, StarNode), 'error: parent is not a StarNode'
        self.parent = parent
        
    def set_left_child(self, left_child):
        'set a left child'
        assert isinstance(left_child, StarNode), 'error: left_child is not a StarNode'
        self.left_child = left_child
        
    def set_right_child(self, right_child):
        'set a right child'
        assert isinstance(right_child, StarNode), 'error: right_child is not a StarNode'
        self.right_child = right_child
        
    def set_sibling(self, sibling):
        'set sibling of a StarNode'
        assert isinstance(sibling, StarNode), 'error: sibling is not a StarNode'
        self.sibling = sibling
        
    def set_ID(self, ID):
        'set ID of this StarNode'
        assert ID < 0, 'error: invalid ID'
        self.ID = ID
        
    def has_parent(self):
        'check if this StarNode has a parent or not'
        if self.parent is not None:
            result = True
        else:
            result = False
        return result
        
    def has_sibling(self):
        'check if this StarNode has a sibling or not'
        if self.sibling is not None:
            result = True
        else:
            result = False
        return result
        
    def has_child(self):
        'check if this StarNode has a child or not'
        if (self.left_child is not None) or (self.right_child is not None):
            result = True
        else:
            result = False
        return result
        
    def has_Ineq(self):
        'check if this StarNode has Inequality part'
        if self.Ineq is not None:
            return True
        else:
            return False
        
    def get_parent(self):
        'get parent of this StarNode'
        return self.parent
    
    def get_left_child(self):
        'get left child of this StarNode'
        return self.left_child
        
    def get_right_child(self):
        'get right child of this StarNode'
        return self.right_child
        
    def get_ID(self):
        'get ID of this StarNode'
        return self.ID
        
    def get_loc(self):
        'get location of this StarNode'
        return self.loc
    
    def get_Eq(self):
        'get Equality part of this StarNode'
        return self.Eq
        
    def get_Ineq(self):
        'get Inequality part of this StarNode'
        return self.Ineq
        
    def get_all_Ineqs(self):
        'get all Inequalities of this StarNode'
        # A StarNode inherits constraints from its accestors
        
        Ineqs = None
        if self.has_Ineq():
            Ineqs = self.get_Ineq()
            if self.has_parent():
                parent_Ineqs = self.parent.get_all_Ineqs()
                if parent_Ineqs is not None:     
                    Ineqs = np.concatenate((Ineqs, parent_Ineqs))
        else:
            if self.has_parent():
                Ineqs = self.parent.get_all_Ineqs()
        return Ineqs
        
    def is_feasible(self):
        'check if this StarNode is feasible or not'
        
        Ineqs = self.get_all_Ineqs()
        n = Ineqs.shape[1]
        c = np.zeros((1, n-1))
        A = Ineqs[:, 0:n-1]
        b = Ineqs[:, n-1]
        
        res = lp(c, A, b)
        if res.status == 0:
            return True
        else:
            return False
            
class StarTree(object):
    'a StarTree symbolic execution of a network'
    
    def __init__(self, depth):
        
        assert depth >= 0, 'error: invalid depth'
        tree = []
        for i in range(depth + 1):
            nodes = []
            tree.append(nodes)
        self.tree = tree
        self.depth = depth        
        
    def add_nodes(self, depth_id, nodes):
        'add some StarNodes into this StarTree'
        
        assert depth_id >= 0, 'error: invalid depth idex'
        for i in range(len(nodes)):
            assert isinstance(nodes[i], StarNode), 'error: {} th object is not a StarNode'.format(i)

        self.tree.insert(depth_id, nodes)
        
    def get_nodes(self, depth_id):
        'get all StarNodes at specific depth'
        return self.tree[depth_id]
        
    def get_leaf_nodes(self):
        'get all leaf nodes'
        return self.get_nodes(self.depth)  
            
class Test(object):
    'test this module'
    
    def test_StarNode():
        'test StarNode'
    
        # create a StarNode that has two children 
        # Root Node: [x; y] = [0; 0] + [1;0]*x + [0; 1]*y
        # LelfChild: [x; y], x >= 0: 
        # RightChild: [0; y], x <= 0 
    
        Eq = np.array([[0, 1, 0], [0, 0, 1]])
        Ineq = None
        # Ineq = np.array([[0, 0, 0]])
    
        root = StarNode(Eq, Ineq)
        Ineq_lc = np.array([[-1, 0, 0]])
        Eq_rc = np.array([[0, 0, 0], [0, 0, 1]])
        Ineq_rc = np.array([[1, 0, 0]])
    
        left_child = StarNode(Eq, Ineq_lc)
        right_child = StarNode(Eq_rc, Ineq_rc)
        if hasattr(left_child, 'set_sibling'):
            print('left_child has set_sibling attribute')
    
        left_child.set_parent(root)
        right_child.set_parent(root)
    
        root.set_left_child(left_child)
        root.set_right_child(right_child)
    
        print('left_child parameters:')
        print('Eq = :', left_child.get_Eq())
        print('Ineqs = :', left_child.get_all_Ineqs())
    
        print('right_child parameters:')
        print('Eq = :', right_child.get_Eq())
        print('Ineqs = :', right_child.get_all_Ineqs())
    
        Ineq_rc2 = np.array([[0, -1, 0]])
        left_child2 = StarNode(Eq, Ineq_rc2)
        left_child2.set_parent(left_child)
    
        print('left_child2 parameters:')
        print('Eq = :', left_child2.get_Eq())
        print('Ineq = :', left_child2.get_all_Ineqs())
    
        if left_child2.is_feasible():
            print('left_child2 is feasible')
        else:
            print('left_child2 is not feasible')
        
        root.print()
    

           
if __name__ == "__main__":
    
    Test.test_StarNode()            
    	
    
	
			
		
		

	