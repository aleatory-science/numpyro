# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from numpy.testing import assert_allclose
from pytest import fail

from jax import numpy as jnp, random, value_and_grad, grad
from jax.scipy.special import logsumexp

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
    def model(obs):
        z = numpyro.sample("z", dist.Normal(0, 1))
        numpyro.sample("obs", dist.Normal(z, 1), obs=obs)

    def guide(x):
        x = numpyro.param('x', 0.)
        numpyro.sample("z", dist.Normal(x, 1))

    def stein_loss_fn(particles, obs):
        return SteinLoss(
            elbo_num_particles=1, stein_num_particles=3
        )._single_draw_particle_loss(
            random.PRNGKey(0),
            particles,
            model,
            guide,
            unravel_pytree,
            (obs,),
            {},
            {},
        )

    xs = jnp.array([-1, 0.5, 3.0])
    m = xs.shape[0]
    ps = {"x": xs}
    zs = jnp.array([-0.90424156, 1.4490576, 2.5476568]) - xs # from inspect
    obs = 2.

    flat_ps, unravel_pytree, _ = batch_ravel_pytree(ps, nbatch_dims=1)
    act_loss, act_grad = value_and_grad(stein_loss_fn)(flat_ps, obs)

    def exp_loss(xs):
        loss = 0.
        for i in range(m):
            z = zs[i] + xs[i]  # Normal.sample uses loc + eps*scale
            lp_m = dist.Normal().log_prob(z) + dist.Normal(z).log_prob(obs)
            lp_g = logsumexp(dist.Normal(xs).log_prob(z)) - jnp.log(m)
            loss += (lp_m - lp_g) / m
        return loss

    assert_allclose(act_loss, exp_loss(xs))

    assert_allclose(act_grad.squeeze(), grad(exp_loss)(xs))
