# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

""" 
SMI implements
1. setup (__init__)
2. init
3. evaluate
4. run
5. update: takes current SMI state and returns the state and loss after one step.
6. get_params: takes the current SMI state and model inputs and returns all parameters in constraint space.

Optimization is done in unconstraint space
"""
from jax import jacrev, vmap
from collections import namedtuple

SMIState = namedtuple("SMIState", ["optim_state", "rng_key"])
SMIRunResult = namedtuple("SMIRunResult", ["params", "state", "losses"])

class SMIGuide:
    def __init__(self, guide, n_particles):
        self.guide = guide
        self.n_particles = n_particles
    
    def __call__(self, *args, **kwargs):
        pass

    def _find_init_params(self, particle_seed, inner_guide, model_args, model_kwargs):
        def local_trace(key):
            guide = deepcopy(inner_guide)

            with handlers.seed(rng_seed=key), handlers.trace() as mixture_trace:
                guide(*model_args, **model_kwargs)

            init_params = {
                name: site["value"]
                for name, site in mixture_trace.items()
                if site.get("type") == "param"
            }
            return init_params

        return vmap(local_trace)(random.split(particle_seed, self.num_stein_particles))

def _make_loss_fn():
    def loss_fn():
        return 0.
    return loss_fn

class SMI:
    def __init__(self, model, guide, n_particles, loss, optim, kernel):
        self.model = model
        self.loss = loss
        self.guide = SMIGuide(guide, n_particles)
        self.kernel = kernel
        self.optim = optim
    
    def init(self):
        pass
        
    def update(self, state: SMIState, *args, **kwargs):
        pass

    def get_params(self, state: SMIState):
        return self.constrain_fn(self.optim.get_params(state.optim_state))
    
    def evaluate(self):
        pass