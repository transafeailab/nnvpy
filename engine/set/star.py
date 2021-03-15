"""
Star set and its methods
Dung Tran: 11/2/2020 Update: 
"""

import numpy as np
from scipy.optimize import linprog as lp
import polytope as pc


class Star(object):
    "Star object"

    def __init__(self, Eq, Ineqs):

        # A star node has two parts:
        # 	1) Equality part: x = c + a1 * v1 + a2 * v2 + ...+ am * vm: Eq = [v1 v2 ... vm c]
        #   2) Inequality part: C * a <= d: Ineq = [C d]

        assert isinstance(Eq, np.ndarray), "error: equality is not an ndarray"

        if isinstance(Ineqs, np.ndarray):
            if Ineqs.shape[1] != Eq.shape[1]:
                raise Exception(
                    "Inconsistency between Equality and Inequality")
        self.Eq = Eq
        self.Ineqs = Ineqs
        self.Dim = Eq.shape[0]
        self.Nvars = Eq.shape[1] - 1
        if Ineqs is None:
            self.Nconstrs = 0
        else:
            self.Nconstrs = Ineqs.shape[0]
        self.lb = None
        self.ub = None

    def get_Eq(self):
        "get Eq of this Star"
        return self.Eq

    def get_Ineqs(self):
        "get inequality of this Star"
        return self.Ineqs

    def get_Dim(self):
        "get dimension of this Star"
        return self.Dim

    def get_Nvars(self):
        "get number of predicate variable of this Star"
        return self.Nvars

    def get_Nconstrs(self):
        "get number of constraints of this Star"
        return self.Nconstrs

    def is_empty(self):
        "check feasibility of this Star"
        Ineqs = self.get_Ineqs()
        if Ineqs is None:
            return False
        else:
            n = self.get_Nvars()
            A = Ineqs[:, 0:n]
            b = Ineqs[:, n]
            c = np.zeros((1, A.shape[1]))
            c[0, 0] = 1
            rs = lp(c, A_ub=A, b_ub=b, method='simplex')
            print(rs)
            if rs.status == 0:
                return False
            else:
                return True

    def set_bounds(self, lb, ub):
        'set lower bound and upper bound'

        assert isinstance(lb,
                          np.ndarray), 'error: lower bound is not an ndarray'
        assert isinstance(ub,
                          np.ndarray), 'error: upper bound is not an ndarray'
        assert lb.shape[0] == ub.shape[
            0], 'error: inconsistency between lower bound and upper bound'
        assert lb.shape[0] == self.Dim, 'error: dimension inconsistency'
        assert lb.shape[1] == 1, 'error: lower bound should be a vector'
        assert ub.shape[1] == 1, 'error: upper bound should be a vector'
        self.lb = lb
        self.ub = ub

    def get_min(self, id):
        'get min of a state'
        assert id >= 0, 'error: invalid index'

        if self.lb is None:
            n = self.get_Nvars()
            Eq = self.get_Eq()
            c = Eq[id, 0:n]
            Ineqs = self.get_Ineqs()
            A = Ineqs[:, 0:n]
            b = Ineqs[:, n]
            res = lp(c, A, b)
            if res.status == 0:
                return res.fun + Eq[id, n]
            else:
                return None
        else:
            return self.lb[id]

    def get_max(self, id):
        'get max of a state'
        assert id >= 0, 'error: invalid index'

        if self.ub is None:
            n = self.get_Nvars()
            Eq = self.get_Eq()
            c = -Eq[id, 0:n]
            Ineqs = self.get_Ineqs()
            A = Ineqs[:, 0:n]
            b = Ineqs[:, n]
            res = lp(c, A, b)
            if res.status == 0:
                return -res.fun + Eq[id, n]
            else:
                return None
        else:
            return self.ub[id]

    def get_lower_bound(self):
        'get the lower bound vector of this Star'

        n = self.get_Dim()
        lb = np.zeros((n, 1))
        for i in range(n):
            lb[i] = self.get_min(i)
        return lb

    def get_specific_lower_bound(self, ids):
        'get the lower bounds at specific states'
        n = len(ids)
        lb = np.zeros((n, 1))
        for i in range(n):
            lb[i] = self.get_min(ids[i])
        return lb

    def get_specific_upper_bound(self, ids):
        'get the lower bounds at specific states'
        n = len(ids)
        ub = np.zeros((n, 1))
        for i in range(n):
            ub[i] = self.get_max(ids[i])
        return ub

    def get_upper_bound(self):
        'get the lower bound vector of this Star'
        n = self.get_Dim()
        ub = np.zeros((n, 1))
        for i in range(n):
            ub[i] = self.get_max(i)
        return ub

    def print(self):
        "print this Star"

        print("======== Star Set Information ========")
        print("*Dimension: ", self.get_Dim())
        print("*Number of predicates: ", self.get_Nvars())
        print("*Number of constraints: ", self.get_Nconstrs())
        print("*Equality: ")
        print(self.get_Eq())
        print("*Inequality: ")
        print(self.get_Ineqs())

    def __str__(self):
        string_repr = ""
        string_repr += "======== Star Set Information ========\n"
        string_repr += "*Dimension: {}\n".format(self.get_Dim())
        string_repr += "*Number of predicates: {}\n".format(self.get_Nvars())
        string_repr += "*Number of constraints: {}\n".format(
            self.get_Nconstrs())
        string_repr += "*Equality:\n{}\n".format(self.get_Eq())
        string_repr += "*Inequality:\n{}\n".format(self.get_Ineqs())
        return string_repr

    def __eq__(self, other):
        # Check if other is a Star
        if not isinstance(other, Star):
            return False
        # Check whether Eq are close
        if not np.allclose(self.get_Eq(), other.get_Eq()):
            return False
        # Handle None cases in InEq
        if self.get_Ineqs() is None or other.get_Ineqs() is None:
            return self.get_Ineqs() is None and other.get_Ineqs() is None
        # Finally, equivalent if InEq are close
        return np.allclose(self.get_Ineqs(), other.get_Ineqs())

    @staticmethod
    def rand(dim, nvars, nconstr):
        "randomly generate a star set"

        assert dim > 0, "error: dimension should be larger than zero"
        assert nvars > 0, "error: number of predicates should be larger than zero"
        assert nconstr > 0, "error: number of constraints should be larger than zero"

        Eq = np.random.rand(dim, dim + 1)
        Ineqs = np.random.rand(nconstr, nvars + 1)
        S = Star(Eq, Ineqs)
        if S.is_empty():
            return Star.rand(dim, nvars, nconstr)
        else:
            return S

    def affine_map(self, W, b):
        'affine mapping of a star is another star'

        assert isinstance(
            W, np.ndarray), 'error: mapping matrix should be an ndarray'
        assert W.shape[
            1] == self.Dim, 'error: inconsistency between the mapping matrix and the Star'
        if b is None:
            Eq = np.dot(W, self.get_Eq())
            return Star(Eq, self.get_Ineqs())
        else:
            if isinstance(b, np.ndarray):
                if b.shape[0] != W.shape[0]:
                    raise Exception(
                        'Inconsistency between mapping matrix and mapping (offset) vector'
                    )
                else:
                    Eq = np.dot(W, self.get_Eq()) + b
                    return Star(Eq, self.get_Ineqs())
            else:
                raise Exception('offset vector is not an ndarray')


class Test(object):
    "Test for Star module"

    def test_Star():
        "test a Star set"

        # construction
        S = Star.rand(3, 3, 2)
        S.print()
        if S.is_empty():
            print("S is an empty set")
        else:
            print("S is not an empty set")

        # min, max, bounds
        a = S.get_min(0)
        print('Min of state {} is: {}'.format(0, a))
        b = S.get_max(0)
        print('Max of state {} is: {}'.format(0, b))

        lb = S.get_lower_bound()
        print('lower bound vector of S: ', lb)
        ub = S.get_upper_bound()
        print('upper bound vector of S: ', ub)

        lb1 = S.get_specific_lower_bound([1, 2])
        print('specific lower bound vector of S: ', lb1)
        ub1 = S.get_specific_upper_bound([1, 2])
        print('specific upper bound vector of S: ', ub1)

        # affine mapping
        W = np.random.rand(2, 3)
        b = np.random.rand(2, 1)
        S1 = S.affine_map(W, b)
        S1.print()
        S2 = S.affine_map(W, None)
        S2.print()


if __name__ == "__main__":

    Test.test_Star()
