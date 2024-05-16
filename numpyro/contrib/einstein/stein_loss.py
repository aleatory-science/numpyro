# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from jax import numpy as jnp, random, vmap, grad
from jax.nn import logsumexp

from numpyro.contrib.einstein.stein_util import batch_ravel_pytree
from numpyro.handlers import replay, seed, trace, substitute
from numpyro.infer.util import log_density
from numpyro.util import _validate_model, check_model_guide_match


class SteinLoss:
    def __init__(self, elbo_num_particles=1, stein_num_particles=1):
        self.elbo_num_particles = elbo_num_particles
        self.stein_num_particles = stein_num_particles
    
    def particle_score(
        self, 
        rng_key, 
        particle,  # particle index to take gradient wrt
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

        # E_q(theta|phi_l) [grad_l (log q(theta|phi_l)) * log {p(theta|model) / (sum_j q(theta|phi_j) }]
        def comp_model_grad(gkey):
            seeded_guide = seed(guide, gkey)
            gtr_l = trace(substitute(seeded_guide, data={**param_map, **unravel_pytree(particle)})).get_trace(*model_args, **model_kwargs)
            grad_lpg_l = grad(lambda par: log_density(
                seeded_guide,
                model_args,
                model_kwargs,
                {**param_map, **unravel_pytree(par)},
            )[0])(particle)

            def clp_fn(par_j):
                clp, ctr = log_density(
                    replay(guide, gtr_l),
                    model_args,
                    model_kwargs,
                    {**param_map, **par_j},
                )
                # Validate
                check_model_guide_match(ctr, gtr_l)
                return clp

            glp = logsumexp(vmap(clp_fn)(ps)) # computes q(theta_l| phi_j)

            seeded_model = seed(model, model_key)
            corr_plates = {
                k: v
                for k, v in trace(seeded_model)
                .get_trace(*model_args, **model_kwargs)
                .items()
                if v["type"] == "plate"
                if jnp.shape(v["value"]) != jnp.shape(gtr_l[k]["value"])
            }
            gtr_l.update(corr_plates)

            mlp, mtr = log_density(
                replay(seeded_model, gtr_l),
                model_args,
                model_kwargs,
                {**param_map, **unravel_pytree(particle)},
            )

            check_model_guide_match(mtr, gtr_l)
            _validate_model(mtr, plate_warning="loose")
            return  grad_lpg_l * (mlp - glp)


        def comp_guide_grad(gkey, par_i):
            def tmp_fn_name(key):
                seeded_guide = seed(guide, key)
                gtr_i = trace(substitute(seeded_guide, data={**param_map, **par_i})).get_trace(*model_args, **model_kwargs)

                def clp_fn(par_j):
                    clp, ctr = log_density(
                        replay(guide, gtr_i),
                        model_args,
                        model_kwargs,
                        {**param_map, **par_j},
                    )
                    # Validate
                    check_model_guide_match(ctr, gtr_i)
                    return jnp.exp(clp)

                glp = vmap(clp_fn)(ps).sum() # computes q(theta_i| phi_j)

                grad_pg_l = grad(lambda par: jnp.exp(log_density(
                    replay(guide, gtr_i),
                    model_args,
                    model_kwargs,
                    {**param_map, **unravel_pytree(par)},
                )[0]))(particle)

                return grad_pg_l / glp
            return vmap(tmp_fn_name)(random.split(gkey, self.elbo_num_particles)).mean()

        mgrad = vmap(comp_model_grad)(random.split(guide_key, self.elbo_num_particles)).mean(0)

        ggrad = vmap(comp_guide_grad)(guide_keys, ps).sum()

        return mgrad - ggrad


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

            glp = vmap(clp_fn)(ps) # computes q(theta_i| phi_j)

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
                {**param_map, **par_i},
            )

            check_model_guide_match(mtr, gtr_i)
            _validate_model(mtr, plate_warning="loose")
            return mlp, glp 

        mlps, glps = vmap(comp_elbo)(guide_keys, ps)

        return mlps - (logsumexp(glps, axis=0) - jnp.log(self.stein_num_particles))


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
        return -elbos.mean(0)
