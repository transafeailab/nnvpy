'''
Test gurobipy
'''

import numpy as np
import gurobipy as gp
from gurobipy import GRB

def test_LP():
    
    Q = np.diag([1, 2, 3])
    f = np.array([[1, 1, 1]])
    A = np.array([[1, 2, 3], [1, 1, 0]])
    b = np.array([4, 1])
    # suppress all consolv output from Gurobi
    # link: https://support.gurobi.com/hc/en-us/articles/360044784552-How-do-I-suppress-all-console-output-from-Gurobi-
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as m:
            x = m.addMVar(3, ub=1.0)
            m.setObjective(f @ x)
            m.addConstr(A@x >= b)
            m.optimize()
            for v in m.getVars():
                print('%s %g' %(v.varName, v.x)) # print all variable names and values
            print('Obj: %g' % m.objVal) # print objective value
  
    

if __name__=='__main__':
    test_LP()
