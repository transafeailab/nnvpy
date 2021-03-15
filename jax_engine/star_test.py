import unittest
import hypothesis

import hypothesis.extra.numpy as npst
from hypothesis.strategies._internal.numbers import integers
from hypothesis.strategies._internal.strategies import T
import numpy as np
from hypothesis import given
import hypothesis.strategies as st

from engine.set.star import Star as OldStar
from jax_engine.star import Star

MIN_NDIM = 1
MAX_NDIM = 4
MIN_NVAR = 1
MAX_NVAR = 4
MIN_NCONSTR = 1
MAX_NCONSTR = 4


def old_star_from_star(star):
    Eqs = np.concatenate((star.basis, np.expand_dims(star.center, 0)), axis=1)
    if star.constraints is not None:
        InEqs = np.concatenate(
            (star.constraints, np.expand_dims(star.upper_bounds, 0)), axis=1)
    else:
        InEqs = None
    return OldStar(Eqs, InEqs)


@st.composite
def star_strategy(draw):
    ndim = draw(st.integers(min_value=MIN_NDIM, max_value=MIN_NDIM))
    nvars = draw(st.integers(min_value=MIN_NVAR, max_value=MIN_NVAR))
    nconstr = draw(st.integers(min_value=MIN_NCONSTR, max_value=MIN_NCONSTR))

    basis = draw(npst.arrays(np.float32, [ndim, nvars]))
    center = draw(npst.arrays(np.float32, ndim))

    star = Star(basis, center)
    return star


@st.composite
def affine_star_strategy(draw):
    """Assumes the dimensions of W (mul) and b (shift) are valid w.r.t. the star"""
    star = draw(star_strategy())
    newdim = draw(st.integers(min_value=MIN_NDIM, max_value=MIN_NDIM))
    mul = draw(npst.arrays(np.float32, [star.dim, newdim]))
    shift = draw(npst.arrays(np.float32, newdim))
    return star, mul, shift


class TestStarEquivalence(unittest.TestCase):
    """Test equivalence to old star"""
    @given(star_strategy())
    def test_basic_props(self, star):
        old_star = old_star_from_star(star)

        # dim
        assert star.dim == old_star.get_Dim(
        ), "star dim {} != old_star dim {} with Eq shape {}".format(
            star.dim, old_star.get_Dim(),
            old_star.get_Eq().shape)

        # nvars
        assert star.nvars == old_star.get_Nvars(
        ), "star nvars {} != old_star nvars {} with Eq shape {}".format(
            star.nvars, old_star.get_Nvars(),
            old_star.get_Eq().shape)

        # nconstr
        assert star.nconstrs == old_star.get_Nconstrs(
        ), "star nvars {} != old_star nvars {} with InEq shape {}".format(
            star.nconstrs, old_star.get_Nconstrs(),
            old_star.get_InEq().shape)

    @given(star_strategy())
    def test_empty(self, star):
        old_star = old_star_from_star(star)
        assert star.is_empty() == old_star.is_empty()

    @given(affine_star_strategy())
    def test_affine_map(self, args):
        star, mul, shift = args
        old_star = old_star_from_star(star)

        star.affine_map(mul, shift)
        affined_old_star = old_star.affine_map(mul, shift)
        assert old_star_from_star(
            star) == affined_old_star, "OLD_STAR:\n{}\n\nSTAR:\n{}".format(
                old_star, star)


if __name__ == "__main__":
    unittest.main()
