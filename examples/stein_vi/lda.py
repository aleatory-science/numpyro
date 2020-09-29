import sys

from jax.experimental import stax
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

import jax
import jax.numpy as jnp
from sklearn.utils import shuffle

import numpyro
import numpyro.distributions as dist

import numpy as np

from numpyro.callbacks import Progbar
from numpyro.contrib.funsor import enum
from numpyro.guides import WrappedGuide
from numpyro.handlers import trace, seed, block, mask
from numpyro.infer import SVI, ELBO, Stein
from numpyro.infer.autoguide import AutoDelta
from numpyro.infer.kernels import RBFKernel
from numpyro.infer.util import _guess_max_plate_nesting
from numpyro.optim import Adam


def lda(doc_words, lengths, num_topics=20, num_words=100, num_max_elements=10,
        num_hidden=100):
    num_docs = doc_words.shape[0]
    topic_word_probs = numpyro.sample('topic_word_probs',
                                      dist.Dirichlet(jnp.ones((num_topics, num_words)) / num_words).to_event(1))
    with numpyro.plate('documents', num_docs, dim=-2):
        document_topic_probs = numpyro.sample('topic_probs', dist.Dirichlet(jnp.ones(num_topics) / num_topics))
        with numpyro.plate('words', num_max_elements, dim=-1):
            word_topic = numpyro.sample('word_topic', dist.Categorical(document_topic_probs))
            indices = jnp.arange(num_max_elements)
            data_mask = indices[None] < lengths[..., None]
            with mask(mask=data_mask):
                numpyro.sample('word', dist.Categorical(topic_word_probs[word_topic]), obs=doc_words)


def lda_guide(doc_words, lengths, num_topics=20, num_words=100, num_max_elements=10,
              num_hidden=100):
    num_docs = doc_words.shape[0]
    topic_word_pcs = numpyro.param('topic_word_pcs', jnp.ones((num_topics, num_words)),
                                   constraint=dist.constraints.positive)
    _topic_word_probs = numpyro.sample('topic_word_probs', dist.Dirichlet(topic_word_pcs).to_event(1))
    amortize_nn = numpyro.module('amortize_nn', stax.serial(
        stax.Dense(num_hidden),
        stax.Relu,
        stax.Dense(num_topics),
        stax.Sigmoid
    ), (num_docs, num_max_elements))
    with numpyro.plate('documents', num_docs, dim=-2):
        document_topic_pcs = amortize_nn(doc_words)[..., None, :] * 1e12 + 1e-12
        _document_topic_probs = numpyro.sample('topic_probs', dist.Dirichlet(document_topic_pcs))


def make_batcher(data, batch_size=32):
    ds_count = data.shape[0] // batch_size
    num_max_elements = np.bincount(data.nonzero()[0]).max()

    def batch_fn(step):
        nonlocal data
        i = step % ds_count
        epoch = step // ds_count
        is_last = i == (ds_count - 1)
        batch_values = data[i * batch_size:(i + 1) * batch_size].todense()
        res = [[] for _ in range(batch_size)]
        for idx1, idx2 in zip(*np.nonzero(batch_values)):
            res[idx1].append(idx2)
        lengths = []
        padded_res = []
        for r in res:
            padded_res.append(r + [0] * (num_max_elements - len(r)))
            lengths.append(len(r))
        if is_last:
            data = shuffle(data)
        return (np.array(padded_res), np.array(lengths)), {}, epoch, is_last

    return batch_fn, num_max_elements


def main(_argv):
    numpyro.enable_validation(True)
    newsgroups = fetch_20newsgroups()['data']
    num_words = 300
    count_vectorizer = CountVectorizer(max_df=.95, min_df=.01,
                                       token_pattern=r'(?u)\b[^\d\W]\w+\b',
                                       max_features=num_words,
                                       stop_words='english')
    newsgroups_docs = count_vectorizer.fit_transform(newsgroups)
    batch_fn, num_max_elements = make_batcher(newsgroups_docs, batch_size=128)
    args, _, _, _ = batch_fn(0)
    rng_key = jax.random.PRNGKey(8938)
    stein = Stein(lda, WrappedGuide(lda_guide),
                  Adam(0.1), ELBO(), RBFKernel(), num_particles=5,
                  num_topics=20, num_words=num_words,
                  num_max_elements=num_max_elements)
    stein.train(rng_key, 100, batch_fun=batch_fn, callbacks=[Progbar()])


if __name__ == '__main__':
    main(sys.argv)
