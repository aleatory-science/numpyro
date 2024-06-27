# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from jax import numpy as jnp, random, vmap
from jax.nn import logsumexp
from jax.lax import stop_gradient

from numpyro.contrib.einstein.stein_util import batch_ravel_pytree
from numpyro.handlers import replay, seed
from numpyro.infer.util import log_density
from numpyro.util import _validate_model, check_model_guide_match


class SteinLoss:
    def __init__(self, elbo_num_particles=1, stein_num_particles=1):
        self.elbo_num_particles = elbo_num_particles
        self.stein_num_particles = stein_num_particles
    
    @staticmethod
    def _make_ld(fn_args, fn_kwargs, unravel_pytree, param_map):
        def ld_fn(fn,p):
            ld, tr =  log_density(
                fn,
                fn_args,
                fn_kwargs,
                {**param_map, **unravel_pytree(p)},
            )
            return ld, tr
        return ld_fn
    

    def _single_draw_particle_loss(
        self,
        rng_key,
        particles,
        model,
        guide,
        unravel_pytree,
        model_args,
        model_kwargs,
        param_map,
    ):

        m = self.stein_num_particles
        keys = random.split(rng_key, m+1)
        gkeys, mkey = keys[:-1], keys[-1]

        ld = self._make_ld(model_args, model_kwargs, unravel_pytree, param_map)

        def comp_elbo(pi, keyi):
            """ Let pi be the i'th of m particles, p(D,t) be the joint model
                and q(t|pi) be the i'th mixture component.
                Computes E_q(t|pi)[log(p(D,t)/(1/m sum_j q(t|pj)))]
            """
            
            _, tri = ld(seed(guide, keyi), pi)

            def comp_ld(pj):
                """ Computes the log density q(t|pj) with t drawn from q(t|pi) """
                ldj, trj = ld(replay(guide, tri),  # Sites drawn wrt pi
                              pj)
                check_model_guide_match(trj, tri)  # Validate guide
                return ldj

            # Score mixture of guides
            ld_g = logsumexp(
                vmap(comp_ld)(particles)
            ) - jnp.log(m)


            # Score model
            ld_m, tr_m = ld(
                replay(seed(model, mkey), tri),
                pi
            )

            # Validation model
            check_model_guide_match(tr_m, tri)
            _validate_model(tr_m, plate_warning="loose")

            return ld_m - ld_g 
        
        all_comp_elbos = vmap(comp_elbo)(particles, gkeys)
        assert all_comp_elbos.shape == (m,)

        return all_comp_elbos.mean()
    
    def particle_loss(
        self,
        rng_key,
        particles,
        model,
        guide,
        unravel_pytree,
        model_args,
        model_kwargs,
        param_map,
    ):
        loss_fn = lambda key: self._single_draw_particle_loss(
            key,
            particles,
            model,
            guide,
            unravel_pytree,
            model_args,
            model_kwargs,
            param_map,
        )

        return vmap(loss_fn)(random.split(rng_key, self.elbo_num_particles)).mean()

    def loss(self, rng_key, param_map, model, guide, particles, *args, **kwargs):
        if not particles:
            raise ValueError("Stein mixture undefined for empty guide.")

        flat_particles, unravel_pytree, _ = batch_ravel_pytree(particles, nbatch_dims=1)

        # NOTE: No longer sampling an assignment!
        return -self.particle_loss(rng_key, flat_particles, model, guide, unravel_pytree, args, kwargs, param_map)