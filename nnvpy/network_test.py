import unittest
import tensorflow as tf

import hypothesis
import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
from hypothesis import given
from jax_engine.star_test import star_strategy

from jax_engine.network import Network

MIN_LAYERS = 1
MAX_LAYERS = 3
MIN_WIDTH = 1
MAX_WIDTH = 5

floats = st.floats(allow_nan=False, allow_infinity=False, width=32)


@st.composite
def network_strategy(draw, input_shape=None):
    net = Network()
    nlayers = draw(st.integers(min_value=MIN_LAYERS, max_value=MAX_LAYERS))

    if input_shape is None:
        prev_width = draw(st.integers(min_value=MIN_WIDTH,
                                      max_value=MAX_WIDTH))
    else:
        prev_width = input_shape
    for _ in range(nlayers):
        width = draw(st.integers(min_value=MIN_WIDTH, max_value=MAX_WIDTH))
        W = draw(npst.arrays(np.float32, [prev_width, width], elements=floats))
        b = draw(npst.arrays(np.float32, [width], elements=floats))
        net.add_dense_layer(W, b)
        prev_width = width
    return net


network_star_strategy = st.integers(
    MIN_WIDTH,
    MAX_WIDTH).flatmap(lambda n: (network_strategy(n), star_strategy(n)))


# Generates mean, covar
@st.composite
def exp_dist_strategy(draw, width=None):
    if width is None:
        width = draw(st.integers(min_value=MIN_WIDTH, max_value=MAX_WIDTH))
    covar_part = draw(npst.arrays(np.float32, [width, width], elements=floats))
    hypothesis.assume(np.any(covar_part))
    covar = covar_part @ covar_part.T
    mean = draw(npst.arrays(np.float32, width, elements=floats))
    return mean, covar


@st.composite
def net_star_dist_strategy(draw):
    width = draw(st.integers(min_value=MIN_WIDTH, max_value=MAX_WIDTH))

    net = draw(network_strategy(width))
    mean, covar = draw(exp_dist_strategy(width))
    return net, mean, covar


class TestNetwork(unittest.TestCase):
    @given(net_star_dist_strategy())
    @hypothesis.settings(max_examples=10)
    def test_astar(self, args):
        net, mean, covar = args
        hypothesis.note("mu: " + str(mean.shape))
        hypothesis.note("layers: " + str([l.W.shape for l in net.layers]))
        net.gaussian_astar(mean, covar)

    def test_tensorflow(self):
        l1 = tf.keras.layers.Dense(5)
        l2 = tf.keras.layers.Dense(1)
        initialize = l2(l1(tf.zeros([1, 4])))

        net = Network()
        net.add_dense_layer(l1.kernel.numpy(), l1.bias.numpy())
        net.add_dense_layer(l2.kernel.numpy(), l2.bias.numpy())

        sigma_part = np.random.normal(size=[4, 4])
        sigma = sigma_part @ sigma_part.T

        stars = net.gaussian_astar(np.zeros([4]), sigma)
        assert np.allclose(np.sum([s.prob for s in stars], 1.))


if __name__ == '__main__':
    unittest.main()