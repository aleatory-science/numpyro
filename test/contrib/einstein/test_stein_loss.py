# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from numpy.testing import assert_allclose
from pytest import fail

from jax import numpy as jnp, random, value_and_grad, vmap
from jax.nn import logsumexp

import numpyro
from numpyro.contrib.einstein.stein_loss import SteinLoss
from numpyro.contrib.einstein.stein_util import batch_ravel_pytree
import numpyro.distributions as dist
from numpyro.infer import Trace_ELBO


def test_single_particle_loss():
    def model(x):
        numpyro.sample("obs", dist.Normal(0, 1), obs=x)

    def guide(x):
        pass

    try:
        SteinLoss(elbo_num_particles=10, stein_num_particles=1).loss(
            random.PRNGKey(0), {}, model, guide, {}, 2.0
        )
        fail()
    except ValueError:
        pass


def test_stein_elbo():
    def model(x):
        numpyro.sample("x", dist.Normal(0, 1))
        numpyro.sample("obs", dist.Normal(0, 1), obs=x)

    def guide(x):
        numpyro.sample("x", dist.Normal(0, 1))

    def elbo_loss_fn(x, param):
        return Trace_ELBO(num_particles=1).loss(
            random.PRNGKey(0), param, model, guide, x
        )

    def stein_loss_fn(x, particles):
        return SteinLoss(elbo_num_particles=1, stein_num_particles=1).loss(
            random.PRNGKey(0), {}, model, guide, particles, x
        )

    elbo_loss, elbo_grad = value_and_grad(elbo_loss_fn)(2.0, {"x": 1.0})
    stein_loss, stein_grad = value_and_grad(stein_loss_fn)(2.0, {"x": jnp.array([1.0])})
    assert_allclose(elbo_loss, stein_loss, rtol=1e-6)
    assert_allclose(elbo_grad, stein_grad, rtol=1e-6)


def test_stein_particle_loss():
    def model(x):
        z = numpyro.sample("z", dist.Normal(0, 1))
        numpyro.sample("obs", dist.Normal(z, 1), obs=x)

    def guide(x):
        z_param = numpyro.param("z_param", 0.0)
        z = numpyro.sample("z", dist.Normal(z_param, 1))
        assert z.shape == ()

    def stein_loss_fn(particles):
        return SteinLoss(elbo_num_particles=1, stein_num_particles=3).mixture_loss(
            random.PRNGKey(0),
            particles,
            model,
            guide,
            (4.0,),
            {},
            unravel_pytree,
            {},
        )

    zps = jnp.array([-1, 0.5, 3.0])
    num_particles = 3
    particles = {"z_param": zps}

    flat_particles, unravel_pytree, _ = batch_ravel_pytree(particles, nbatch_dims=1)
    act_loss = stein_loss_fn(flat_particles)

    z = jnp.array([-0.1241799, 0.84642684, 3.0385242])  # From inpected traces

    def man_loss(zps):
        exp_prior_loss = dist.Normal().log_prob(z)
        exp_like_loss = dist.Normal(z).log_prob(4.0)
        exp_guide_loss = vmap(dist.Normal(zps).log_prob)(z)
        exp_loss = (
            exp_prior_loss
            + exp_like_loss
            - (logsumexp(exp_guide_loss, axis=0) - jnp.log(num_particles))
        )
        return exp_loss.mean()

    assert_allclose(act_loss, man_loss(zps))
