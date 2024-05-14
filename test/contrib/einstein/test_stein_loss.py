# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from numpy.testing import assert_allclose
from pytest import fail

from jax import numpy as jnp, random, value_and_grad, grad, jacrev

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
        z_param = numpyro.param('z_param', 0.)
        z = numpyro.sample("z", dist.Normal(z_param, 1))
        assert z.shape == ()

    def stein_loss_fn(particles):
        return SteinLoss(
            elbo_num_particles=1, stein_num_particles=3
        ).mixture_loss(
            random.PRNGKey(0),
            particles,
            model,
            guide,
            (2.,),
            {},
            unravel_pytree,
            {'z': jnp.array(1.)},
        )

    zs = jnp.array([-1, 0.5, 3.0])
    num_particles = zs.shape[0]
    particles = {"z_param": zs}

    flat_particles, unravel_pytree, _ = batch_ravel_pytree(particles, nbatch_dims=1)
    loss = stein_loss_fn(flat_particles)

    # let t=1 in 1/3 \sum_i=1^3  \log {N(2|t, 1) N(t|0,1)/ [(1/3) N(t|-1, 1) + N(t|0.5, 1) + N(t|3,1)]}
    expected_loss = (dist.Normal().log_prob(1.) + dist.Normal(1.).log_prob(2.)  # joint log density of model
                     + jnp.log(3.) - jnp.log(jnp.exp(dist.Normal(zs).log_prob(jnp.ones_like(zs))).sum())) # log density of guides

    assert_allclose(loss, expected_loss)

    grads = grad(lambda ps: stein_loss_fn(ps))(flat_particles).squeeze()

    qs_sum, grads_loc = value_and_grad(lambda zs: jnp.exp(dist.Normal(zs).log_prob(jnp.ones_like(zs))).sum())(zs)
    expected_grads = - grads_loc / qs_sum
    assert_allclose(expected_grads, grads)  # NOTE: This does not account for gradient wrt. expect prob.
