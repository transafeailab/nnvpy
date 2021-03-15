import numpy as np
from numpy.core.defchararray import upper
from scipy.optimize import linprog as lp
from copy import copy


class Star(object):
    def __init__(self, basis, center, constraints=None, upper_bounds=None):
        """
        A star node has two parts:
          1) Equality part: x = c + a1 * v1 + a2 * v2 + ...+ am * vm: Eq = [v1 v2 ... vm c]
          2) Inequality part: C * a <= d: Ineq = [C d]

        Shapes:
          basis - [ndims, nvars]
          center - [ndim]
          constraints - [nconstraints, nvars]
          upper_bounds - [nconstraints]
        """
        assert len(basis.shape) == 2  # matrix
        assert len(center.shape) == 1  # vector
        assert basis.shape[-1] == center.shape[0]

        if constraints is not None or upper_bounds is not None:
            assert len(constraints.shape) == 2  # matrix
            assert len(upper_bounds.shape) == 1  # vector
            assert constraints.shape[-1] == upper_bounds.shape[0]

            assert center.shape == upper_bounds.shape

        self._basis = basis
        self._center = center
        self._constraints = constraints
        self._ub = upper_bounds

    @property
    def basis(self):
        return self._basis

    @property
    def center(self):
        return self._center

    @property
    def constraints(self):
        return self._constraints

    @property
    def upper_bound(self):
        return self._ub

    @property
    def dim(self):
        return self.basis.shape[0]

    @property
    def nvars(self):
        """number of predicate variables"""
        return self.basis.shape[1]

    @property
    def nconstrs(self):
        """number of constraints"""
        if self.constraints is None:
            return 0
        else:
            return self.constraints.shape[0]

    def is_empty(self):
        """check feasibility"""
        if self.constraints is None:
            return False

        c = np.zeros_like(self.upper_bound)
        c[0] = 1
        rs = lp(c,
                A_ub=self.constraints,
                b_ub=self.upper_bound,
                method='simplex')
        return rs.status != 0

    def affine_map(self, W, b=None, right=True):
        self._basis = np.dot(self._basis, W)
        self._center = np.dot(self._center, W)
        if b is not None:
            self._basis += b
            self._center += b

    def stepRelu(self, index):
        # upper child
        if self.constraints is not None:
            upper_constraints = np.concatenate(
                (self.constraints, -1 * self.basis[index]), axis=0)
            upper_upper_bounds = np.concatenate(
                (self.upper_bound, self.center[index]), axis=0)
        else:
            upper_constraints = -1 * self.basis[index]
            upper_upper_bounds = self.center[index]
        upper_star = Star(np.copy(self.basis), np.copy(self.center),
                          upper_constraints, upper_upper_bounds)

        # lower child
        lower_basis = np.copy(self.basis)
        lower_basis[index, :] = 0
        if self.constraints is not None:
            lower_constraints = np.concatenate(
                (self.constraints, self.basis[index]), axis=0)
            lower_upper_bounds = np.concatenate(
                (self.upper_bound, self.center[index]), axis=0)
        else:
            lower_constraints = self.basis[index]
            lower_upper_bounds = self.center[index]
        lower_star = Star(lower_basis, np.copy(self.center), lower_constraints,
                          lower_upper_bounds)

        return [s for s in [upper_star, lower_star] if not s.is_empty()]

    def __repr__(self):
        string_repr = "Star: "
        string_repr += "*Basis:\n{}\n".format(self.basis)
        string_repr += "*Center:\n{}\n".format(self.center)
        string_repr += "*Constraints:\n{}\n".format(self.constraints)
        string_repr += "*Upper Bounds:\n{}\n".format(self.upper_bound)
        return string_repr

    def __str__(self):
        string_repr = ""
        string_repr += "======== Star Set Information ========\n"
        string_repr += "*Dimension: {}\n".format(self.dim)
        string_repr += "*Number of predicates: {}\n".format(self.nvars)
        string_repr += "*Number of constraints: {}\n".format(self.nconstrs)
        string_repr += "*Basis:\n{}\n".format(self.basis)
        string_repr += "*Center:\n{}\n".format(self.center)
        string_repr += "*Constraints:\n{}\n".format(self.constraints)
        string_repr += "*Upper Bounds:\n{}\n".format(self.upper_bound)
        return string_repr