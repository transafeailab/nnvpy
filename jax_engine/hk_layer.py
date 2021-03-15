import haiku as hk
import jax.numpy as jnp
import numpy as np
from numpy.core.defchararray import startswith

from engine.set import star


def hk_affine_star(self, in_star: star.Star):
    """Wrapper for Haiku Linear layer"""
    inputs = in_star.get_Eq()

    input_size = self.input_size = inputs.shape[-1]
    output_size = self.output_size
    dtype = inputs.dtype

    w_init = self.w_init
    if w_init is None:
        stddev = 1. / np.sqrt(self.input_size)
        w_init = hk.initializers.TruncatedNormal(stddev=stddev)
    w = hk.get_parameter("w", [input_size, output_size], dtype, init=w_init)

    eq_out = jnp.dot(in_star.get_Eq(), w)

    if self.with_bias:
        b = hk.get_parameter("b", [self.output_size], dtype, init=self.b_init)
        b = jnp.broadcast_to(b, eq_out.shape)
        eq_out += b
    return star.Star(np.array(eq_out), in_star.get_Ineqs())


def hk_call_with_star(self, W, b=None):
    if isinstance(W, star.Star):
        return self.affine_star(W)
    else:
        return self(W, b)


hk.Linear.affine_star = hk_affine_star
hk.Linear.__call__ = hk_call_with_star
