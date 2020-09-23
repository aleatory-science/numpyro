import sys

from jax.experimental import stax
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

import jax
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist


def lda(doc_words, num_topics=20):
    num_docs = doc_words.shape[0]
    num_words = doc_words.shape[1]
    topic_word_probs = numpyro.sample('topic_word_probs', dist.Dirichlet(jnp.ones(num_words) / num_words)
                                      .expand_by(num_topics).to_event(1))
    with numpyro.plate('documents', num_docs, dim=-2):
        document_topic_probs = numpyro.sample('topic_probs', dist.Dirichlet(jnp.ones(num_topics) / num_topics))
        with numpyro.plate('words', num_words, dim=-1):
            word_topic = numpyro.sample('word_topic', dist.Categorical(document_topic_probs))
            numpyro.sample('word', dist.Categorical(topic_word_probs[word_topic]), obs=(doc_words >= 1) * 1)


def lda_guide(doc_words, num_topics=20):
    num_docs = doc_words.shape[0]
    num_words = doc_words.shape[1]
    topic_word_pcs = numpyro.param('topic_word_pcs', jnp.ones((num_topics, num_words)),
                                   constraint=dist.constraints.simplex)
    _topic_word_probs = numpyro.sample('topic_word_probs', dist.Dirichlet(topic_word_pcs).to_event(1))
    amortize_nn = numpyro.module('amortize_nn', stax.serial(
        stax.Dense((num_topics + num_words) / 2),
        stax.Relu,
        stax.Dense(num_topics),
        stax.Exp
    ))
    with numpyro.plate('documents', num_docs, dim=-2):
        _document_topic_probs = numpyro.sample('topic_probs', dist.Dirichlet(amortize_nn(doc_words)))


def main(_argv):
    newsgroups = fetch_20newsgroups()['data']
    count_vectorizer = CountVectorizer(max_df=.95, min_df=.05)
    res = count_vectorizer.fit_transform(newsgroups)
    x = 1


if __name__ == '__main__':
    main(sys.argv)
