import jax
import jax.numpy as jnp
from jax.lax import scan
import matplotlib.pyplot as plt
import numpy.random as npr
from jax.config import config

import numpyro
import numpyro.distributions as dist
from numpyro.callbacks import Progbar
from numpyro.infer import SVI, ELBO
from numpyro.infer.autoguide import AutoDelta
from numpyro.optim import Adam


def predator_prey(prey0, predator0, *, r=0.6, k=100,
                  s=1.2, a=25, u=0.5, v=0.3,
                  step_size=0.01, num_iterations=100):
    def pp_upd(state, _):
        prey, predator = state
        sh = prey * predator / (a + prey)
        prey_upd = r * prey * (1 - prey / k) - s * sh
        predator_upd = u * sh - v * predator
        prey = prey + step_size * prey_upd
        predator = predator + step_size * predator_upd
        return (prey, predator), (prey, predator)

    _, (prey, predator) = scan(pp_upd, (prey0, predator0),
                               jnp.arange(step_size, num_iterations + step_size, step_size))
    prey = jnp.reshape(prey, (num_iterations, -1))[:, 0]
    predator = jnp.reshape(predator, (num_iterations, -1))[:, 0]
    return prey, predator


def model(prey, predator, idxs):
    prior = dist.HalfNormal(10.)
    prey0 = numpyro.sample('prey0', prior)
    predator0 = numpyro.sample('predator0', prior)
    r = numpyro.sample('r', prior)
    k = numpyro.sample('k', prior)
    s = numpyro.sample('s', prior)
    a = numpyro.sample('a', prior)
    u = numpyro.sample('u', prior)
    v = numpyro.sample('v', prior)
    sprey, spredator = predator_prey(prey0, predator0,
                                     r=r, k=k, s=s,
                                     a=a, u=u, v=v)
    with numpyro.plate('data', prey.shape[0], dim=-2):
        numpyro.sample('prey_obs', dist.Normal(sprey[idxs], 1.), obs=prey)
        numpyro.sample('predator_obs', dist.Normal(spredator[idxs], 1.), obs=predator)


if __name__ == '__main__':
    config.update('jax_debug_nans', True)
    prey, predator = predator_prey(50., 5.)
    scale = 3
    idxs = [2, 13, 29, 31, 47]
    obs_prey = prey[idxs] + scale * npr.randn(1000, len(idxs))
    obs_predator = predator[idxs] + scale * npr.randn(1000, len(idxs))
    plt.plot(obs_prey.transpose(), color='b')
    plt.plot(obs_predator.transpose(), color='r')
    plt.show()
    guide = AutoDelta(model)
    svi = SVI(model, guide, Adam(0.25), ELBO())
    rng_key = jax.random.PRNGKey(1377)
    svi.train(rng_key, 10_000, obs_prey, obs_predator, idxs, callbacks=[Progbar()])
