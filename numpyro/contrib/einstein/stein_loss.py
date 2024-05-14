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
        guide_key, model_key = random.split(rng_key, 2)
        guide_keys = random.split(guide_key, self.stein_num_particles)

        ps = vmap(unravel_pytree)(particles)

        def comp_elbo(gkey, curr_par):
            seeded_guide = seed(guide, gkey)
            curr_lp, curr_gtr = log_density(
                seeded_guide,
                model_args,
                model_kwargs,
                {**param_map, **curr_par},
            )

            def clp_fn(cpar):
                clp, ctr = log_density(
                    replay(guide, curr_gtr),
                    model_args,
                    model_kwargs,
                    {**param_map, **cpar},
                )
                # Validate
                check_model_guide_match(ctr, curr_gtr)
                return clp

            glp = logsumexp(vmap(clp_fn)(ps)) - jnp.log(self.stein_num_particles)

            seeded_model = seed(model, model_key)
            corr_plates = {
                k: v
                for k, v in trace(seeded_model)
                .get_trace(*model_args, **model_kwargs)
                .items()
                if v["type"] == "plate"
                if jnp.shape(v["value"]) != jnp.shape(curr_gtr[k]["value"])
            }
            curr_gtr.update(corr_plates)

            mlp, mtr = log_density(
                replay(seeded_model, curr_gtr),
                model_args,
                model_kwargs,
                {**param_map, **curr_par},
            )

            check_model_guide_match(mtr, curr_gtr)
            _validate_model(mtr, plate_warning="loose")
            comp_elbo = mlp - glp
            return comp_elbo

        return vmap(comp_elbo, out_axes=0)(guide_keys, ps)

    def loss(self, rng_key, param_map, model, guide, particles, *args, **kwargs):
        if not particles:
            raise ValueError("Stein mixture undefined for empty guide.")

        flat_particles, unravel_pytree, _ = batch_ravel_pytree(particles, nbatch_dims=1)

        score_keys = random.split(rng_key, self.elbo_num_particles)

        elbos = vmap(
            lambda key: self.mixture_loss(
                rng_key=key,
                particles=flat_particles,
                model=model,
                guide=guide,
                model_args=args,
                model_kwargs=kwargs,
                unravel_pytree=unravel_pytree,
                param_map=param_map,
            ),
            out_axes=0,
        )(score_keys)
        return elbos.mean(0)
