import unittest

import hypothesis
import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
from engine.operation.operation import stepReLU
from engine.set.star import Star as OldStar
from hypothesis import given

from jax_engine.star import Star

MIN_NDIM = 1
MAX_NDIM = 4
MIN_NVAR = 1
MAX_NVAR = 4
MIN_NCONSTR = 1
MAX_NCONSTR = 4

floats = st.floats(allow_nan=False, allow_infinity=False, width=32)


def old_star_from_star(star):
    Eqs = np.concatenate((star.basis, np.expand_dims(star.center, -1)), axis=1)
    if star.constraints is not None:
        InEqs = np.concatenate(
            (star.constraints, np.expand_dims(star.upper_bounds, -1)), axis=1)
    else:
        InEqs = None
    return OldStar(Eqs, InEqs)


@st.composite
def star_strategy(draw, input_dim=None):
    if input_dim is None:
        input_dim = draw(st.integers(min_value=MIN_NDIM, max_value=MAX_NDIM))
    repr_dim = draw(st.integers(min_value=MIN_NVAR, max_value=MAX_NVAR))
    nconstr = draw(st.integers(min_value=MIN_NCONSTR, max_value=MAX_NCONSTR))

    basis = draw(
        npst.arrays(np.float32, [input_dim, repr_dim], elements=floats))
    hypothesis.assume(np.any(basis))
    center = draw(npst.arrays(np.float32, input_dim, elements=floats))

    star = Star(basis, center)
    return star


@st.composite
def affine_star_strategy(draw):
    """Assumes the dimensions of W (mul) and b (shift) are valid w.r.t. the star"""
    star = draw(star_strategy())
    newdim = draw(st.integers(min_value=MIN_NDIM, max_value=MAX_NDIM))
    mul = draw(
        npst.arrays(np.float32, [star.repr_dim, newdim], elements=floats))
    shift = draw(npst.arrays(np.float32, newdim, elements=floats))
    return star, mul, shift


@st.composite
def star_valid_index(draw):
    star = draw(star_strategy())
    index = draw(st.integers(min_value=0, max_value=star.repr_dim - 1))
    return star, index


class TestStarEquivalence(unittest.TestCase):
    """Test equivalence to old star"""
    @given(star_strategy())
    def test_basic_props(self, star):
        old_star = old_star_from_star(star)

        # dim
        assert star.repr_dim == old_star.get_Dim(
        ), "star dim {} != old_star dim {} with Eq shape {}".format(
            star.repr_dim, old_star.get_Dim(),
            old_star.get_Eq().shape)

        # nvars
        assert star.input_dim == old_star.get_Nvars(
        ), "star nvars {} != old_star nvars {} with Eq shape {}".format(
            star.input_dim, old_star.get_Nvars(),
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
            star
        ) == affined_old_star, "\nOLD_STAR:\n{}\nSTAR:\n{}\nMUL:\n{}\n\nSHIFT\n{}".format(
            affined_old_star, star, mul, shift)

    @given(star_valid_index())
    def test_step_relu(self, args):
        """This test is limited in that it only tests one step of step_relu and not a sequence"""
        star, index = args

        star_children = star.step_relu(index)
        star_children_old = [
            old_star_from_star(child) for child in star_children
        ]

        stepReLU_object = stepReLU(index)
        old_star = old_star_from_star(star)
        old_star_children = stepReLU_object.execute_single_input(old_star)
        assert [star_children_old == old_star_children], str(star)


if __name__ == "__main__":
    print(old_star_from_star(star_strategy().example()))
    exit()
    unittest.main()
