from stat import S_IFBLK
import haiku as hk
import jax
import jax.numpy as jnp

import jax_engine.hk_layer
from engine.set.star import Star


def softmax_cross_entropy(logits, labels):
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)


network = lambda x: hk.Sequential([
    hk.Linear(300),
    #jax.nn.relu,
    hk.Linear(100),
    #jax.nn.relu,
    hk.Linear(10),
])(x)


def loss_fn(network, images, labels):
    logits = network(images)
    return jnp.mean(softmax_cross_entropy(logits, labels))


def star_fn(star):
    layer = hk.Linear(2)
    out_star = layer(star)
    return out_star


# There are two transforms in Haiku, hk.transform and hk.transform_with_state.
# If our network updated state during the forward pass (e.g. like the moving
# averages in hk.BatchNorm) we would need hk.transform_with_state, but for our
# simple MLP we can just use hk.transform.
loss_fn_t = hk.transform(loss_fn)
star_fn_t = hk.transform(star_fn)

# MLP is deterministic once we have our parameters, as such we will not need to
# pass an RNG key to apply. without_apply_rng is a convenience wrapper that will
# make the rng argument to `loss_fn_t.apply` default to `None`.
loss_fn_t = hk.without_apply_rng(loss_fn_t)
star_fn_t = hk.without_apply_rng(star_fn_t)

# star construction
S = Star.rand(3, 3, 2)
S.print()

# create params
rng = jax.random.PRNGKey(42)
params = star_fn_t.init(rng, S)

# run star forward
out_star = star_fn_t.apply(params, S)
out_star.print()

gradient = jax.grad(loss_fn_t.apply)
star_gradient = jax.grad(star_fn_t.apply)
print(gradient)
print(star_gradient(S))