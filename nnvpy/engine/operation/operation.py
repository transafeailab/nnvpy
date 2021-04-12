'''
Operation module contains all related operations such as affinemapping(AM), stepReLU
Dung Tran: 12/12/2020
'''

import numpy as np
from engine.set.star import Star


class stepReLU(object):
    'stepReLU operation'

    def __init__(self, ID):
        assert ID >= 0, 'error: Invalid index'
        self.ID = ID  # index of this stepReLU operation

    def execute_single_input(self, S):
        'execute a stepReLU operation'

        assert isinstance(S, Star), 'error: input is not a Star'
        assert self.ID <= S.get_Dim(
        ), 'error: Invalid ID for stepReLU operation'
        # x[ID]>=0 -> left_child
        Eq = S.get_Eq()
        Ineqs = S.get_Ineqs()
        n = S.get_Nvars()
        Eq_id = np.array([Eq[self.ID, :]])
        new_Ineq = -np.copy(Eq_id)
        new_Ineq[0, n] = -new_Ineq[0, n]
        if Ineqs is None:
            new_Ineqs = new_Ineq
        else:
            new_Ineqs = np.concatenate((Ineqs, new_Ineq), axis=0)
        S1 = Star(Eq, new_Ineqs)

        Eq2 = np.copy(Eq)
        Eq2[self.ID, :] = 0
        new_Ineq2 = -np.copy(new_Ineq)
        if Ineqs is None:
            new_Ineqs2 = new_Ineq2
        else:
            new_Ineqs2 = np.concatenate((Ineqs, new_Ineq2), axis=0)
        S2 = Star(Eq2, new_Ineqs2)

        a = S1.is_empty()
        b = S2.is_empty()
        rs = []
        if not a:
            rs.append(S1)
        if not b:
            rs.append(S2)
        return rs

    def execute(self, S):
        'execute stepReLU opeartion on multiple Stars'

        assert isinstance(S, list), 'error: input should be a list of stars'
        n = len(S)
        rs = []
        for i in range(n):
            rs = rs + self.execute_single_input(S[i])
        return rs


class AM(object):
    'affine mapping operation'

    def __init__(self, W, b):

        assert isinstance(W,
                          np.ndarray), 'error: weight matrix is not an ndarray'
        assert isinstance(b,
                          np.ndarray), 'error: bias vector is not an ndarray'
        assert W.shape[0] == b.shape[
            0], 'error: inconsistency between weight matrix and bias vector'
        assert b.shape[1] == 1, 'error: bias vector has more than one column'

        self.W = W
        self.b = b

    def execute(self, S):
        'execution of an affine mapping operation'

        assert isinstance(S,
                          list), 'error: input set should be a list of Stars'
        n = len(S)
        rs = []
        for i in range(n):
            assert isinstance(S[i],
                              Star), 'error: input {} is not a Star'.format(i)
            rs.append(S[i].affine_map(self.W, self.b))
        return rs
