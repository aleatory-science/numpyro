
from matplotlib import pyplot as plt

from numpyro import prng_key, set_platform, sample
from numpyro.contrib.einstein import RBFKernel, SteinVI, MixtureGuidePredictive
from numpyro.contrib.einstein.stein_loss import NewSteinLoss
from numpyro.handlers import seed
from numpyro.infer.autoguide import AutoDelta, AutoNormal
from numpyro.infer import SVI, Trace_ELBO
from numpyro.distributions import Gamma, Normal
from numpyro.optim import Adam, Adagrad
from jax import numpy as jnp, jacrev
set_platform('gpu')

from functools import partial
from tqdm import tqdm

from jax import numpy as jnp, vmap, grad, jit, vjp, random
from numpyro.distributions import Normal, Gamma
from numpyro.handlers import seed
from copy import copy
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
from random import shuffle

# SteinVI imports
from numpyro import sample, plate, prng_key
from numpyro.contrib.einstein import RBFKernel, SteinVI
from numpyro.contrib.einstein.steinvi import SteinVIState
from numpyro.infer import Trace_ELBO, SVI
from numpyro.infer.autoguide import AutoDelta
from numpyro.optim import SGD
from numpyro.contrib.einstein.stein_util import (
    batch_ravel_pytree, median_bandwidth
)


dim = 1
num_particles = 3
steps=1
p = Normal(jnp.zeros(dim), jnp.ones(dim)).to_event(1)
log_p = p.log_prob

with seed(rng_seed=1):
    particles = p.sample(prng_key(), (num_particles,))

def rbf_kernel(x, y, bandwidth):
    return jnp.exp(-0.5 * ((x - y) ** 2).sum() / bandwidth)


def all_pairs_sq_dists(xs):
    pairwise_sq_dist = jnp.maximum(
        jnp.sum(xs**2, 1)[None, :]
        + jnp.sum(xs**2, 1)[:, None]
        - 2 * jnp.matmul(xs, xs.T),
        0.0,
    )
    return pairwise_sq_dist


def median_trick(xs):
    return .5 * jnp.median(all_pairs_sq_dists(xs)) / jnp.log1p(xs.shape[0]) + jnp.finfo(xs.dtype).eps


def attr_force(x, xs):
    return vmap( 
        lambda y: rbf_kernel(
                x, y, bandwidth=median_trick(particles)
            ) 
            * grad(log_p)(y)
        )(xs).mean(0)

def s1(sps, bandwidth):
    return vmap(
        lambda y: vmap(
            lambda x: rbf_kernel(
                x, y, bandwidth=bandwidth
            )  # using cached bandwidth from above cell!
            * grad(log_p)(x)
        )(sps).mean(0)
    )(sps)

def score(sps):
    return vmap(grad(log_p))(sps)
        
loss = NewSteinLoss(1, num_particles)

def model():
    sample('x', Normal())

    
guide = AutoDelta(model)
stein_particles, unravel_pytree, unravel_pytree_batched = batch_ravel_pytree(
            {'x_auto_loc': particles}, nbatch_dims=1
        )

def kernel_particles_loss_fn(
        key, particles
    ):  # TODO: rewrite using def to utilize jax caching
        grads = jacrev(
                lambda ps: loss.mixture_loss(
                    rng_key=key,
                    particles=ps,
                    model=model,
                    guide=guide,
                    unravel_pytree=unravel_pytree,
                    model_args=(),
                    model_kwargs={},
                    param_map={},
                ))(particles).sum(0)
        return grads

print(particles)

print(score(particles))
with seed(rng_seed=0):
    print(kernel_particles_loss_fn(prng_key(), particles))
