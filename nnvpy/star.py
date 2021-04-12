import numpy as np
from scipy.optimize import linprog as lp
from truncnorm import mv_normal_cdf


class Star():
    def __init__(self,
                 basis: np.array,
                 center: np.array,
                 constraints: np.array = None,
                 upper_bounds: np.array = None,
                 is_right_mul: bool = True):
        """
        A star node has two parts:
          1) Equality part: x = c + a1 * v1 + a2 * v2 + ...+ am * vm: Eq = [v1 v2 ... vm c]
          2) Inequality part: C * a <= d: Ineq = [C d]

        Shapes (reverse basis dims for left mul):
          basis - [input_dim, repr_dim]
          center - [repr_dim]
          constraints - [num_constraints, input_dim]
          upper_bounds - [num_constraints, 1]
        """
        assert len(basis.shape) == 2  # matrix
        assert len(center.shape) == 1  # vector
        if is_right_mul:
            assert basis.shape[1] == center.shape[0]
        else:
            assert basis.shape[0] == center.shape[1]

        if constraints is not None or upper_bounds is not None:
            assert len(constraints.shape) == 2, "shape: {}".format(
                constraints.shape)  # matrix
            assert len(upper_bounds.shape) == 1, "shape: {}".format(
                upper_bounds.shape)  # vector
            # assert constraints.shape[-1] == upper_bounds.shape[0]

        self._basis = basis
        self._center = center
        self._constraints = constraints
        self._ub = upper_bounds
        self.is_right_mul = is_right_mul

    @property
    def basis(self) -> np.ndarray:
        return self._basis

    @property
    def center(self) -> np.ndarray:
        return self._center

    @property
    def constraints(self) -> np.ndarray:
        return self._constraints

    @property
    def upper_bounds(self) -> np.ndarray:
        return self._ub

    @property
    def repr_dim(self) -> int:
        if not self.is_right_mul:
            return self.basis.shape[0]
        else:
            return self.basis.shape[1]

    @property
    def input_dim(self) -> int:
        """number of input dimensions"""
        if not self.is_right_mul:
            return self.basis.shape[1]
        else:
            return self.basis.shape[0]

    @property
    def nconstrs(self) -> int:
        """number of constraints"""
        if self.constraints is None:
            return 0
        elif self.is_right_mul:
            return self.constraints.shape[0]
        else:
            return self.constraints.shape[1]

    def is_empty(self) -> bool:
        """check feasibility"""
        if self.constraints is None:
            return False

        c = np.zeros((1, self.constraints.shape[1]))
        c[0, 0] = 1
        rs = lp(c,
                A_ub=self.constraints,
                b_ub=self.upper_bounds,
                method='simplex')
        return rs.status != 0

    def affine_map(self, W, b=None):
        """
        Args:
            W: multiply matrix
            b: shift vector
        """
        if self.is_right_mul:
            self._basis = np.dot(self._basis, W)
            self._center = np.dot(self._center, W)
        else:
            self._basis = np.dot(W, self._basis)
            self._center = np.dot(W, self._center)
        if b is not None:
            self._center += b

    def step_relu(self, index):
        assert self.is_right_mul  # only implemented for now
        # upper child
        new_constraint = np.expand_dims(-1 * self.basis[:, index], 0)
        new_upper_bound = np.expand_dims(self.center[index], 0)
        if self.constraints is not None:
            upper_constraints = np.concatenate(
                (self.constraints, new_constraint), axis=0)
            upper_upper_bounds = np.concatenate(
                (self.upper_bounds, new_upper_bound), axis=0)
        else:
            upper_constraints = new_constraint
            upper_upper_bounds = new_upper_bound
        upper_star = Star(np.copy(self.basis), np.copy(self.center),
                          upper_constraints, upper_upper_bounds)

        # lower child
        lower_basis = np.copy(self.basis)
        lower_basis[0] = 0
        new_constraint = np.expand_dims(self.basis[:, index], 0)
        new_upper_bound = np.expand_dims(self.center[index], 0)
        if self.constraints is not None:
            lower_constraints = np.concatenate(
                (self.constraints, new_constraint), axis=0)
            lower_upper_bounds = np.concatenate(
                (self.upper_bounds, new_upper_bound), axis=0)
        else:
            lower_constraints = new_constraint
            lower_upper_bounds = new_upper_bound
        lower_star = Star(lower_basis, np.copy(self.center), lower_constraints,
                          lower_upper_bounds)

        return [s for s in [upper_star, lower_star] if not s.is_empty()]

    def trunc_gaussian_cdf(self, mu, sigma, n=10000):
        if self.is_right_mul:
            sigma_star = self.constraints @ sigma @ self.constraints.T
            ub = self.upper_bounds - mu @ self.constraints.T
        else:
            sigma_star = self.constraints.T @ self.basis @ self.constraints
            ub = self.upper_bounds - mu @ self.constraints
        lb = np.ones_like(ub)
        lb.fill(np.NINF)
        prob_est, prob_rel_err, prob_up_bound = mv_normal_cdf(
            lb, ub, sigma_star + 1e-10 * np.eye(sigma_star.shape[0]), n)
        return prob_est, prob_up_bound

    def get_max(self, id):
        'get max in one dimension'
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

    def __repr__(self):
        string_repr = "Star: "
        string_repr += "*Basis:\n{}\n".format(self.basis)
        string_repr += "*Center:\n{}\n".format(self.center)
        string_repr += "*Constraints:\n{}\n".format(self.constraints)
        string_repr += "*Upper Bounds:\n{}\n".format(self.upper_bounds)
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
        string_repr += "*Upper Bounds:\n{}\n".format(self.upper_bounds)
        return string_repr
