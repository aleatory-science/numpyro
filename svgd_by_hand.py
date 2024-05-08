
from matplotlib import pyplot as plt

from numpyro import prng_key, set_platform, sample
from numpyro.contrib.einstein import RBFKernel, SteinVI, MixtureGuidePredictive
from numpyro.handlers import seed
from numpyro.infer.autoguide import AutoDelta, AutoNormal
from numpyro.infer import SVI, Trace_ELBO
from numpyro.distributions import Gamma, Normal
from numpyro.optim import Adam, Adagrad
from jax import numpy as jnp
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
num_particles = 1_000
steps = 5_000
p = Normal(jnp.zeros(dim), jnp.ones(dim)).to_event(1)
log_p = p.log_prob

with seed(rng_seed=1):
    particles = p.sample(prng_key(), (num_particles,))

# Uses exact form of https://github.com/DartML/Stein-Variational-Gradient-Descent/blob/master/python/svgd.py#L10-L14
# Note this is different from the reported med^2/log(n) reported in the paper!
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

def svgd_median_trick(xs):
    sq_dist = pdist(xs)
    pairwise_dists = squareform(sq_dist)**2
    h = np.median(pairwise_dists)  
    h = np.sqrt(0.5 * h / np.log(xs.shape[0]+1)) **2
    return h


def median_trick(xs):
    return .5 * jnp.median(all_pairs_sq_dists(xs)) / jnp.log1p(xs.shape[0]) + jnp.finfo(xs.dtype).eps



bandwidth = median_trick(particles)
print("manual", bandwidth)
print('svgd', svgd_median_trick(particles))


def attr_force(x, xs):
    return vmap( 
        lambda y: rbf_kernel(
                x, y, bandwidth=median_trick(particles)
            ) 
            * grad(log_p)(y)
        )(xs).mean(0)
    
def rep_force(x,xs, bandwidth):
    return vmap(grad(lambda y: rbf_kernel(x, y, bandwidth=bandwidth)))(xs).mean(0)

def s1(xs):
    return vmap(
        partial(attr_force, xs=xs)
    )(xs)


def s2(xs):
    bandwidth = median_trick(xs)
    return vmap(partial(rep_force, xs=xs, bandwidth=bandwidth))(xs)


def s1_resample(sps, tps, bandwidth):
    return vmap(
        lambda y: vmap(
            lambda x: rbf_kernel(
                x, y, bandwidth=bandwidth
            )  # using cached bandwidth from above cell!
            * grad(lambda x_: log_p(x_))(x)
        )(tps).mean(0)
    )(sps)

    

def update(ps, wrt_qs, epsilon):
    return ps + epsilon * (
                    jit(s1_resample)(ps, wrt_qs, median_trick(wrt_qs)) + jit(s2)(ps)
                )

""" Without resampling """
def svgd(num_steps, draw_q0, num_particles=20, epsilon=0.1, start_from=None):
    with seed(rng_seed=0):
        ps = [draw_q0() if start_from is None else start_from]
        for _ in tqdm(range(1, num_steps)):
            ps.append(update(ps[-1], ps[-1], epsilon))
        return jnp.array(ps)

svgd_part = svgd(steps, draw_q0=lambda : p.sample(prng_key(), (num_particles,)))

def train_stein(guide, model):
    kernel = RBFKernel()
    stein = SteinVI(
        model,
        guide,
        SGD(1e-1),
        kernel,
        num_stein_particles=num_particles,
        num_elbo_particles=1,
    )


    with seed(rng_seed=0):
        res = stein.run(prng_key(), steps)

    return stein, res

    
def model():
    sample('x', Normal())

    
guide = AutoDelta(model)
ie, res = train_stein(guide, model)
with seed(rng_seed=1):
    _, smi_grads = ie._svgd_loss_and_grads(prng_key(), ie.optim.get_params(res.state.optim_state))

smi_particles = res.params['x_auto_loc'].reshape(-1,1)

svgd_smi_part = svgd(steps, draw_q0=lambda : smi_particles)

x = jnp.linspace(svgd_part.min(), svgd_part.max()).reshape(-1,1)

plt.plot(x, jnp.exp(log_p(x)))
plt.hist([svgd_part[-1].squeeze(), res.params['x_auto_loc'], svgd_smi_part[-1].squeeze(), particles.squeeze()], density=True, bins=50)
plt.legend(['density', 'svgd', 'smi', 'smi_corrected', 'true'])
plt.show()

