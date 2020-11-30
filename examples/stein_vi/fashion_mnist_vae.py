import sys
from math import sqrt
from random import randint

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import stax

import numpyro
from numpyro import distributions as dist
from numpyro.callbacks import Progbar, History, EarlyStopping, TerminateOnNaN
from numpyro.callbacks.reduce_lr import ReduceLROnPlateau
from numpyro.examples.datasets import load_dataset, FASHION_MNIST
from numpyro.infer import SVI, ELBO, Predictive
from numpyro.optim import Adam

import matplotlib.pyplot as plt


def Reshape(new_shape):
    def init_fun(_rng_key, input_shape):
        batch_size, *rest_shape = input_shape
        assert np.prod(new_shape) == np.prod(rest_shape)
        return (batch_size, *new_shape), ()

    def apply_fun(_params, inputs, **_kwargs):
        return jnp.reshape(inputs, (-1, *new_shape))

    return init_fun, apply_fun


def guide(mnist_like=None, latent_dim=30, batch_size=1, data_sizes=(28, 28), eps=1e-3):
    if mnist_like is not None:
        batch_size, *data_sizes = mnist_like.shape
    else:
        mnist_like = jnp.zeros((batch_size, *data_sizes))
    encoder = numpyro.module('encoder',
                             stax.serial(stax.Conv(32, (3, 3), (2, 2), padding='SAME'),
                                         stax.Relu,
                                         stax.BatchNorm(),
                                         stax.Conv(32, (3, 3), (2, 2), padding='SAME'),
                                         stax.Relu,
                                         stax.BatchNorm(),
                                         stax.Conv(64, (3, 3), (2, 2), padding='SAME'),
                                         stax.BatchNorm(),
                                         stax.Relu,
                                         stax.Flatten,
                                         stax.Dense(latent_dim * 2)),
                             input_shape=(batch_size, *data_sizes, 1))
    with numpyro.plate('data', batch_size):
        mnist_like = jnp.reshape(mnist_like, (batch_size, *data_sizes, 1))
        enc_params = jnp.reshape(encoder(mnist_like), (batch_size, latent_dim, 2))
        mus = enc_params[..., 0]
        sigmas = jnp.exp(enc_params[..., 1])
        _zs = numpyro.sample('zs', dist.Normal(mus, sigmas).to_event(1))


def model(mnist_like=None, latent_dim=30, batch_size=1, data_sizes=(28, 28), eps=1e-3):
    if mnist_like is not None:
        batch_size, *data_sizes = mnist_like.shape
    decoder = numpyro.module('decoder',
                             stax.serial(stax.Dense(7 * 7 * 32),
                                         stax.Relu,
                                         Reshape((7, 7, 32)),
                                         stax.ConvTranspose(64, (3, 3), (2, 2), padding='SAME'),
                                         stax.Relu,
                                         stax.BatchNorm(),
                                         stax.ConvTranspose(2, (3, 3), (2, 2), padding='SAME')),
                             input_shape=(batch_size, latent_dim))
    with numpyro.plate('data', batch_size):
        zs = numpyro.sample('zs', dist.Normal(0, 1).expand_by((latent_dim,)).to_event(1))
        decoded = decoder(zs)
        mu = numpyro.deterministic('xs_mu', decoded[..., 0])
        sigma = numpyro.deterministic('xs_sigma', jnp.exp(decoded[..., 1]))
        numpyro.sample('xs', dist.Normal(mu, sigma).to_event(2), obs=mnist_like)


def _plot_mnist_like(images):
    rc = int(sqrt(len(images)))
    rr = len(images) // rc
    fig, ax = plt.subplots(nrows=rr, ncols=rc)
    for i in range(rr):
        for j in range(rc):
            ax[i, j].axis('off')
            ax[i, j].imshow(images[rc * i + j])
    plt.show()


def _make_batcher():
    init, get_batch = load_dataset(FASHION_MNIST, 32)
    num_batches, idxs = init()
    test_batch, _ = get_batch(idxs=idxs)

    def batch_fun(step):
        i = step % num_batches
        epoch = step // num_batches
        is_last = i == (num_batches - 1)
        batch, _ = get_batch(i, idxs)
        return (batch,), {}, epoch, is_last

    return batch_fun, test_batch


def _plot_predictive(rng_key, params, num_samples=32):
    predictive = Predictive(model, params=params, num_samples=num_samples)
    sample_images = jnp.clip(jnp.reshape(predictive(rng_key)['xs'], (num_samples, 28, 28)), 0, 1)
    _plot_mnist_like(sample_images)


def _plot_history(history):
    plt.plot(history.training_history)
    plt.show()


def main(_argv):
    numpyro.set_platform('gpu')
    numpyro.enable_validation()
    num_steps = 25_000
    rng_key = jax.random.PRNGKey(randint(0, 10000))
    rng_key, pred_rng_key = jax.random.split(rng_key)
    svi = SVI(model, guide, Adam(1e-3), ELBO())
    batch_fun, test_batch = _make_batcher()

    _plot_mnist_like(test_batch)
    history = History()
    svi_state, loss = svi.train(rng_key, num_steps, batch_fun=batch_fun,
                                callbacks=[Progbar(), ReduceLROnPlateau(), TerminateOnNaN(), history])
    _plot_predictive(pred_rng_key, svi.get_params(svi_state))
    _plot_history(history)


if __name__ == '__main__':
    main(sys.argv)
