# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from jax import numpy as jnp, random, vmap
from jax.nn import logsumexp

from numpyro.contrib.einstein.stein_util import batch_ravel_pytree
from numpyro.handlers import replay, seed, trace
from numpyro.infer.util import log_density
from numpyro.util import _validate_model, check_model_guide_match


class SteinLoss:
    def __init__(self, elbo_num_particles=1, stein_num_particles=1):
        self.elbo_num_particles = elbo_num_particles
        self.stein_num_particles = stein_num_particles

    def mixture_loss(
        self,
        rng_key,
        particles,
        model,
        guide,
        model_args,
        model_kwargs,
        unravel_pytree,
        param_map,
    ):
        def _single_draw_loss(draw_key):
            """Compute the Stein loss for a single draw"""

            guide_key, model_key = random.split(draw_key, 2)
            guide_keys = random.split(guide_key, self.stein_num_particles)

            ps = vmap(unravel_pytree)(particles)

            def comp_elbo(gkey, par_i):
                seeded_guide = seed(guide, gkey)
                curr_lp, gtr_i = log_density(
                    seeded_guide,
                    model_args,
                    model_kwargs,
                    {**param_map, **par_i},
                )

                def clp_fn(par_j):
                    clp, ctr = log_density(
                        replay(guide, gtr_i),
                        model_args,
                        model_kwargs,
                        {**param_map, **par_j},
                    )
                    # Validate
                    check_model_guide_match(ctr, gtr_i)
                    return clp

                glp = vmap(clp_fn)(ps)  # computes q(theta_i| phi_j)

                seeded_model = seed(model, model_key)
                corr_plates = {
                    k: v
                    for k, v in trace(seeded_model)
                    .get_trace(*model_args, **model_kwargs)
                    .items()
                    if v["type"] == "plate"
                    if jnp.shape(v["value"]) != jnp.shape(gtr_i[k]["value"])
                }
                gtr_i.update(corr_plates)

                mlp, mtr = log_density(
                    replay(seeded_model, gtr_i),
                    model_args,
                    model_kwargs,
                    {**param_map},
                )

                check_model_guide_match(mtr, gtr_i)
                _validate_model(mtr, plate_warning="loose")
                return mlp, glp

            mlps, glps = vmap(comp_elbo)(guide_keys, ps)

            return mlps - (logsumexp(glps, axis=0) - jnp.log(self.stein_num_particles))

        return vmap(_single_draw_loss)(
            random.split(rng_key, self.elbo_num_particles)
        ).mean()

    def loss(self, rng_key, param_map, model, guide, particles, *args, **kwargs):
        if not particles:
            raise ValueError("Stein mixture undefined for empty guide.")

        flat_particles, unravel_pytree, _ = batch_ravel_pytree(particles, nbatch_dims=1)
        stein_loss = self.mixture_loss(
            rng_key=rng_key,
            particles=flat_particles,
            model=model,
            guide=guide,
            model_args=args,
            model_kwargs=kwargs,
            unravel_pytree=unravel_pytree,
            param_map=param_map,
        )
        return -stein_loss
