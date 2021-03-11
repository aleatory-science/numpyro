import argparse
import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import Trace_ELBO, SVI, NUTS, MCMC
from numpyro.infer.autoguide import AutoIAFNormal
from numpyro.infer.reparam import NeuTraReparam


def exp_quad_kernel(x, marginal_std, length_scale, noise_scale):
    if jnp.ndim(x) > 1:
        norms = jnp.linalg.norm(x[None, :, :] - x[:, None, :])
    else:
        norms = x[None, :] - x

    cov = marginal_std * jnp.exp(-.5 * jnp.power(norms / length_scale, 2))
    return cov + jnp.eye(x.shape[0]) * noise_scale


def model(features, labels):
    """ Gaussian process """
    length_scale = numpyro.sample('length_scale', dist.InverseGamma(5, 5))
    marginal_std = numpyro.sample('marginal_std', dist.HalfNormal())
    noise_scale = numpyro.sample('noise_scale', dist.HalfNormal())
    cov = exp_quad_kernel(features, marginal_std, length_scale, noise_scale)
    numpyro.sample('obs', dist.MultivariateNormal(covariance_matrix=cov), obs=labels)


def infer_neutra(data, args, rng_key):
    svi_key, mcmc_key = random.split(rng_key)
    guide = AutoIAFNormal(model, num_flows=args.num_flows)
    svi = SVI(model, guide, numpyro.optim.Adam(0.001), Trace_ELBO())
    params, losses = svi.run(svi_key, args.num_steps, *data)
    plt.plot(losses)
    plt.show()

    neutra = NeuTraReparam(guide, params)
    kernel = NUTS(neutra.reparam(model))
    mcmc = MCMC(kernel, args.num_warmup, args.num_samples)
    mcmc.run(mcmc_key, *data)
    mcmc.print_summary()


def gen_data(rng_key, n, d, marginal_std=1., length_scale=1., noise_scale=.1):
    features_key, obs_key = random.split(rng_key)

    features = dist.Normal(0., 2.).sample(features_key, (n, d))
    cov = exp_quad_kernel(features, marginal_std=marginal_std, length_scale=length_scale, noise_scale=noise_scale)
    obs = dist.MultivariateNormal(0., cov).sample(obs_key)
    return features, obs


def main(args):
    data_key, rng_key = random.split(random.PRNGKey(args.seed))

    features, obs = gen_data(data_key, args.num_data, args.num_dim)

    if args.flow_method == 'neutra':
        infer_neutra((features, obs), args, rng_key)


if __name__ == "__main__":
    assert numpyro.__version__.startswith('0.5.0')
    parser = argparse.ArgumentParser(description="Gaussian Process Example")
    parser.add_argument("-n", "--num-samples", nargs="?", default=100, type=int)
    parser.add_argument("--num-warmup", nargs='?', default=100, type=int)
    parser.add_argument("--num-flows", nargs='?', default=1, type=int)
    parser.add_argument("--num-steps", nargs='?', default=1_000, type=int)
    parser.add_argument("--num-chains", nargs='?', default=1, type=int)
    parser.add_argument("--seed", nargs='?', default=37, type=int)
    parser.add_argument("--num_data", type=int, default=100)
    parser.add_argument("--num_dim", type=int, default=2)
    parser.add_argument("--device", default='cpu', type=str, choices=['cpu', 'gpu'])
    parser.add_argument("--x64", action="store_true")
    parser.add_argument("--flow_method", default='neutra', choices=['neutra'])
    parser.add_argument("--disable-progbar", action="store_true")
    args = parser.parse_args()

    if args.device == "cpu":
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    numpyro.enable_x64(args.x64)
    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)
