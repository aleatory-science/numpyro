# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Example: Hamiltonian Monte Carlo with Energy Conserving Subsampling
===================================================================

This example illustrates the use of data subsampling in HMC using Energy Conserving Subsampling. Data subsampling
is applicable when the likelihood factorizes as a product of N terms.

**References:**

    1. *Hamiltonian Monte Carlo with energy conserving subsampling*,
       Dang, K. D., Quiroz, M., Kohn, R., Minh-Ngoc, T., & Villani, M. (2019)

.. image:: ../_static/img/examples/hmcecs.png
    :align: center
"""

import argparse
import time

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import sklearn.metrics as metrics
from collections import defaultdict

from jax import random,vmap
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.examples.datasets import HIGGS, load_dataset
from numpyro.infer import HMC, HMCECS, MCMC, NUTS, SVI, Trace_ELBO, autoguide, Predictive
from numpyro import handlers


def model(data, subsample_size,obs=None):
    n, m = data.shape
    theta = numpyro.sample('theta', dist.Normal(jnp.zeros(m), .5 * jnp.ones(m)))
    with numpyro.plate('N', n, subsample_size=subsample_size):
        batch_feats = numpyro.subsample(data, event_dim=1)
        batch_obs = numpyro.subsample(obs, event_dim=0)
        numpyro.sample('obs', dist.Bernoulli(logits=theta @ batch_feats.T), obs=batch_obs)


def run_hmcecs(hmcecs_key, args, data, obs, inner_kernel):
    svi_key, mcmc_key = random.split(hmcecs_key)

    # find reference parameters for second order taylor expansion to estimate likelihood (taylor_proxy)
    optimizer = numpyro.optim.Adam(step_size=1e-3)
    guide = autoguide.AutoDelta(model)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    params, losses = svi.run(svi_key, args.num_svi_steps, data,args.subsample_size,obs)
    ref_params = {'theta': params['theta_auto_loc']}

    # taylor proxy estimates log likelihood (ll) by
    # taylor_expansion(ll, theta_curr) +
    #     sum_{i in subsample} ll_i(theta_curr) - taylor_expansion(ll_i, theta_curr) around ref_params
    proxy = HMCECS.taylor_proxy(ref_params)

    kernel = HMCECS(inner_kernel, num_blocks=args.num_blocks, proxy=proxy)
    mcmc = MCMC(kernel, num_warmup=args.num_warmup, num_samples=args.num_samples)

    mcmc.run(mcmc_key, data, args.subsample_size,obs)
    summary_dict = mcmc.print_summary()
    return losses, mcmc.get_samples(),summary_dict


def run_hmc(mcmc_key, args, data, obs, kernel):
    mcmc = MCMC(kernel, num_warmup=args.num_warmup, num_samples=args.num_samples)
    mcmc.run(mcmc_key, data, None,obs)
    summary_dict = mcmc.print_summary()
    return mcmc.get_samples(),summary_dict

def main(args):
    assert 11_000_000 >= args.num_datapoints, "11,000,000 data points in the Higgs dataset"
    # full dataset takes hours for plain hmc!
    if args.dataset == 'higgs':
        _, fetch = load_dataset(HIGGS, shuffle=False, num_datapoints=args.num_datapoints)
        data, obs = fetch()
        data_train, data_test, obs_train, obs_test = data[:100], data[100:120], obs[:100], obs[100:120]
    else:
        data, obs = (np.random.normal(size=(20, 28)), np.ones(20))
        data_train,data_test,obs_train,obs_test = data[:16], data[16:], obs[:16],obs[16:]


    hmcecs_key1,hmcecs_key2,hmcecs_key3, hmc_key, pred_key = random.split(random.PRNGKey(args.rng_seed),5)

    # choose inner_kernel
    if args.inner_kernel == 'hmc':
        inner_kernel = HMC(model)
    else:
        inner_kernel = NUTS(model)
    sampling_keys = [hmcecs_key1,hmcecs_key2,hmcecs_key3]
    hmcecs_dict_samples = {}
    for i, rnd_key in zip(list(range(3)),sampling_keys):
        print("HMC-ECS run number {}".format(i))
        start = time.time()

        losses, hmcecs_samples,hmcecs_sum_dict = run_hmcecs(rnd_key, args, data_train, obs_train, inner_kernel)
        hmcecs_runtime = time.time() - start
        hmcecs_dict_samples["Run_{}".format(i)] = hmcecs_samples["theta"]

    start = time.time()
    hmc_samples,hmc_sum_dict = run_hmc(hmc_key, args, data_train, obs_train, inner_kernel)
    hmc_runtime = time.time() - start
    convergence_plot(hmc_samples,hmcecs_dict_samples)
    exit()
    summary_plot(losses, hmc_samples, hmcecs_samples[0], hmc_runtime, hmcecs_runtime)
    # TODO: Fix predictions, handler not working
    #predicted_labels = make_predictions(data_test,hmcecs_samples,args.num_samples,pred_key)
    exit()
    predicted_labels=np.array([[0,1,1,1],[0,1,0,1],[0,1,1,1]])
    roc_curve(predicted_labels,obs_test)

def convergence_plot(hmc_samples,hmcecs_samples):
    n = len(hmcecs_samples) +1
    colors = cm.rainbow(np.linspace(0,1,n))
    for i,(run,samples) in enumerate(hmcecs_samples.items()):
        plt.plot(list(range(samples.shape[0])),samples[:,0],color=colors[i],label="HMC-ECS {}".format(i))
    plt.plot(list(range(hmc_samples["theta"].shape[0])),hmc_samples["theta"][:,0], color=colors[-1],label="HMC")
    plt.legend()
    plt.title("Convergence plot")
    plt.savefig("convergence.pdf")

def summary_plot(losses, hmc_samples, hmcecs_samples, hmc_runtime, hmcecs_runtime):
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(losses, 'r')
    ax[0, 0].set_title('SVI losses')
    ax[0, 0].set_ylabel('ELBO')

    if hmc_runtime > hmcecs_runtime:
        ax[0, 1].bar([0], hmc_runtime, label='hmc', color='b')
        ax[0, 1].bar([0], hmcecs_runtime, label='hmcecs', color='r')
    else:
        ax[0, 1].bar([0], hmcecs_runtime, label='hmcecs', color='r')
        ax[0, 1].bar([0], hmc_runtime, label='hmc', color='b')
    ax[0, 1].set_title('Runtime')
    ax[0, 1].set_ylabel('Seconds')
    ax[0, 1].legend()
    ax[0, 1].set_xticks([])

    ax[1, 0].plot(jnp.sort(hmc_samples['theta'].mean(0)), 'or')
    ax[1, 0].plot(jnp.sort(hmcecs_samples['theta'].mean(0)), 'b')
    ax[1, 0].set_title(r'$\mathrm{\mathbb{E}}[\theta]$')

    ax[1, 1].plot(jnp.sort(hmc_samples['theta'].var(0)), 'or')
    ax[1, 1].plot(jnp.sort(hmcecs_samples['theta'].var(0)), 'b')
    ax[1, 1].set_title(r'Var$[\theta]$')

    for a in ax[1, :]:
        a.set_xticks([])

    fig.tight_layout()
    fig.savefig('hmcecs_summary_plot.pdf', bbox_inches='tight')

def make_predictions(test_data,samples,num_samples,key):
    #TODO: Fix , handler not working
    # def predict(model, rng_key, samples, *args, **kwargs):
    #     model = handlers.substitute(handlers.seed(model, rng_key), samples)
    #     # note that Y will be sampled in the model because we pass Y=None here
    #     model_trace = handlers.trace(model).get_trace(*args, **kwargs)
    #     return model_trace['obs']['value']
    # vmap_args = (samples, random.split(random.PRNGKey(1), num_samples))
    # predictions = vmap(lambda samples, rng_key: predict(model, rng_key, samples, test_data,args.subsample_size))(*vmap_args)
    # return predictions
    pred_fn = Predictive(model,samples,num_samples=num_samples)

    pred = pred_fn(key,test_data,args.subsample_size)


def roc_curve(predicted_labels,test_labels):
    "Plot the Receiver Operander Characteristic"

    predicted_labels = np.argmax(predicted_labels.T,axis=1)
    fpr,tpr,_ = metrics.roc_curve(test_labels,predicted_labels)
    roc_auc = metrics.auc(fpr, tpr)

    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    # method I: plt
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic HIGGS dataset')
    plt.legend(loc='lower right', prop={'size': 6})
    plt.savefig('hmcecs_ROC.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hamiltonian Monte Carlo with Energy Conserving Subsampling")
    parser.add_argument('--subsample_size', type=int, default=2) #1300
    parser.add_argument('--num_svi_steps', type=int, default=5000) #5000
    parser.add_argument('--num_blocks', type=int, default=100) #100
    parser.add_argument('--num_warmup', type=int, default=500) #500
    parser.add_argument('--num_samples', type=int, default=500) #500
    parser.add_argument('--num_datapoints', type=int, default=1_500_000)
    parser.add_argument('--dataset', type=str, choices=['higgs', 'mock'], default='mock')
    parser.add_argument('--inner_kernel', type=str, choices=['nuts', 'hmc'], default='nuts')
    parser.add_argument('--device', default='cpu', type=str, choices=['cpu', 'gpu'])
    parser.add_argument('--rng_seed', default=37, type=int, help='random number generator seed')

    args = parser.parse_args()

    numpyro.set_platform(args.device)

    main(args)
