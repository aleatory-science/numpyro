#%%

from tqdm import tqdm

from numpyro.util import safe_mul, safe_div

import jax
from jax import config
config.update("jax_debug_nans", True)
import jax.ops
import jax.numpy as jnp
from functools import partial

import numpyro
from numpyro import distributions as dist
from numpyro.contrib.autoguide import AutoDelta
from numpyro.examples.runge_kutta import runge_kutta_4
from numpyro.infer import SVI, ELBO
from numpyro.optim import Adam

#%%

rng_key = jax.random.PRNGKey(242)

#%% md

## Predator Prey Model

#%%

def predator_prey_step(t, state, r=0.6, k=100, s=1.2, a=25, u=0.5, v=0.3):
    prey = state[..., 0]
    predator = state[..., 1]
    sh = safe_div(safe_mul(prey, predator), a + prey)
    prey_upd = r * safe_mul(prey, safe_div(1 - prey, k)) - s * sh
    predator_upd = u * sh - v * predator
    return jnp.stack((prey_upd, predator_upd), axis=-1)
num_time = 5
step_size = 0.1
num_steps = int(num_time / step_size)
dampening_rate = 0.9
lyapunov_scale = 1e-3
clip = lambda x: jnp.clip(x, -10.0, 10.0)
predator_prey = runge_kutta_4(predator_prey_step, step_size, num_steps, dampening_rate,
                              lyapunov_scale, clip,
                              unconstrain_fn=lambda _, x: jnp.where(x < 10, jnp.log(jnp.expm1(x)), x),
                              constrain_fn=lambda _, x: jax.nn.softplus(x))
predator_prey = partial(predator_prey, rng_key)

#%%

indices = jnp.array([1, 11, 21, 31, 41])
res, lyapunov_loss = predator_prey(jnp.array([50., 5.]))
# res = np.reshape(res, (num_time, num_steps // num_time, -1))[:, 0, :]
noise = jax.random.normal(rng_key, (1000,5,2)) * 10
data = (indices, res[indices] + noise)
data

#%%

def model(indices, observations):
    prior_dist = dist.HalfNormal(1000)
    prey0 = numpyro.sample('prey0', prior_dist)
    predator0 = numpyro.sample('predator0', prior_dist)
    r = numpyro.sample('r', prior_dist)
    k = numpyro.sample('k', prior_dist)
    s = numpyro.sample('s', prior_dist)
    a = numpyro.sample('a', prior_dist)
    u = numpyro.sample('u', prior_dist)
    v = numpyro.sample('v', prior_dist)
    ppres, lyapunov_loss = predator_prey(jnp.array([prey0, predator0]), r=r, k=k, s=s, a=a, u=u, v=v)
    # ppres = np.reshape(ppres, (num_time, num_time // num_steps, -1))
    numpyro.factor('lyapunov_loss', lyapunov_loss)
    numpyro.sample('obs', dist.Normal(ppres[indices], 10.0).to_event(2), obs=observations)

#%% md

### SVI

#%%

svi = SVI(model, AutoDelta(model), Adam(0.1), ELBO())
state = svi.init(rng_key, *data)
pbar = tqdm(range(27))
prev_state = state
for i in pbar:
    prev_state = state
    state, loss = svi.update(state, *data)
    pbar.set_description(f'SVI {loss}')

#%%

_, rng_key_debug = jax.random.split(prev_state.rng_key)
params = svi.optim.get_params(prev_state.optim_state)
grad_fun = lambda params, *data: jax.value_and_grad(lambda x: svi.loss.loss(rng_key_debug, svi.constrain_fn(x),
                                                    svi.model, svi.guide, *data))(params)

#%%

with jax.disable_jit():
    svi.evaluate(prev_state, *data)
