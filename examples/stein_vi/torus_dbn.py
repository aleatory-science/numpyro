import os
import re
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import tqdm

from numpyro.contrib.control_flow import scan
from numpyro.distributions.transforms import AffineTransform
from numpyro.handlers import mask, trace, replay
from .multiple_formatter import Multiple
from .protein_parser import ProteinParser


def torus_dbn(max_length, phis=None, psis=None, lengths=None,
              num_sequences=None, num_states=55,
              prior_conc=0.1, prior_loc=0.0,
              prior_length_shape=100., prior_length_rate=100.,
              prior_kappa_min=10., prior_kappa_max=1000.):
    # From https://github.com/pyro-ppl/numpyro/blob/master/examples/hmm_enum.py
    if lengths is not None:
        assert num_sequences is None
        num_sequences = lengths.shape[0]
    else:
        assert num_sequences is not None
    transition_probs = numpyro.sample('transition_probs',
                                      dist.Dirichlet(jnp.ones(num_states, num_states, dtype='float')
                                                     * num_states).to_event(1))
    length_shape = numpyro.sample('length_shape', dist.HalfCauchy(prior_length_shape))
    length_rate = numpyro.sample('length_rate', dist.HalfCauchy(prior_length_rate))
    phi_locs = numpyro.sample('phi_locs',
                              dist.VonMises(jnp.ones(num_states, dtype='float') * prior_loc,
                                            jnp.ones(num_states, dtype='float') * prior_conc).to_event(1))
    phi_kappas = numpyro.sample('phi_kappas', dist.Uniform(jnp.ones(num_states, dtype='float') * prior_kappa_min,
                                                           jnp.ones(num_states, dtype='float') * prior_kappa_max
                                                           ).to_event(1))
    psi_locs = numpyro.sample('psi_locs',
                              dist.VonMises(jnp.ones(num_states, dtype='float') * prior_loc,
                                            jnp.ones(num_states, dtype='float') * prior_conc).to_event(1))
    psi_kappas = numpyro.sample('psi_kappas', dist.Uniform(jnp.ones(num_states, dtype='float') * prior_kappa_min,
                                                           jnp.ones(num_states, dtype='float') * prior_kappa_max
                                                           ).to_event(1))
    with numpyro.plate('sequences', num_sequences, dim=-2):
        if lengths is not None:
            obs_length = lengths.float().unsqueeze(-1)
        else:
            obs_length = None
        sam_lengths = numpyro.sample('length',
                                     dist.TransformedDistribution(dist.GammaPoisson(length_shape, length_rate),
                                                                  AffineTransform(1., 0.)), obs=obs_length)
        if lengths is None:
            lengths = sam_lengths.squeeze(-1).long()

        def transition_fn(carry, phi, psi):
            prev_state, t = carry
            with mask(mask=(t < lengths)[..., None]):
                state = numpyro.sample('state', dist.Categorical(transition_probs[prev_state]))
                if phi is not None:
                    obs_phi = phi[..., None]
                else:
                    obs_phi = None
                if psi is not None:
                    obs_psi = psi[..., None]
                else:
                    obs_psi = None
                with numpyro.plate('elements', 1, -1):
                    numpyro.sample('phi', dist.VonMises(phi_locs[state], phi_kappas[state]), obs=obs_phi)
                    numpyro.sample('psi', dist.VonMises(psi_locs[state], psi_kappas[state]), obs=obs_psi)

        state_init = jnp.zeros((num_sequences, 1), dtype='int32')
        phis = jnp.swapaxes(phis, 0, 1) if phis is not None else [None] * max_length
        psis = jnp.swapaxes(psis, 0, 1) if psis is not None else [None] * max_length
        scan(transition_fn, (state_init, 0), (phis, psis))


def infer_discrete(get_trace, first_available_dim):
    def run(*args, **kwargs):
        return {}
    return run


def sample_and_plot(model, guide, filename=None, num_sequences=128, num_states=5):
    guide_trace = trace(guide).get_trace(num_sequences=num_sequences, num_states=num_states)
    samples = infer_discrete(trace(replay(model, guide_trace)).get_trace,
                             first_available_dim=-3)(num_sequences=num_sequences,
                                                     num_states=num_states)
    lengths = samples['length']['value'][..., 0].int()
    phis = jnp.concatenate([site['value'] for name, site in samples
                            if re.match(r'phi_\d', name)], dim=-1)
    psis = jnp.cat([site['value'] for name, site in samples
                    if re.match(r'psi_\d', name)], dim=-1)
    plot_rama(lengths, phis, psis, filename=filename)


def plot_rama(lengths, phis, psis, filename='rama', dir='figs'):
    fig, ax = plt.subplots()
    ax.hexbin(np.concatenate([phiseq[:lengths[t]].flatten() for t, phiseq in enumerate(phis.detach().numpy())]),
              np.concatenate([psiseq[:lengths[t]].flatten() for t, psiseq in enumerate(psis.detach().numpy())]),
              bins='log')
    multiple = Multiple()
    ax.set_xlabel(r'$\phi$')
    ax.set_ylabel(r'$\psi$')
    ax.xaxis.set_major_formatter(multiple.formatter())
    ax.xaxis.set_major_locator(multiple.locator())
    ax.yaxis.set_major_formatter(multiple.formatter())
    ax.yaxis.set_major_locator(multiple.locator())
    os.makedirs(dir, exist_ok=True)
    fig.savefig(os.path.join(dir, f'{filename}.png'))
    plt.close(fig)


def plot_losses(total_losses, filename='elbo', dir='figs'):
    fig, ax = plt.subplots()
    ax.plot(total_losses)
    os.makedirs(dir, exist_ok=True)
    fig.savefig(os.path.join(dir, f'{filename}.png'))
    plt.close(fig)


def main(_argv):
    aas, ds, phis, psis, lengths = ProteinParser.parsef_jnp('data/TorusDBN/top500.txt')
    guide = AutoDelta(poutine.block(torus_dbn, hide_fn=lambda site: site['name'].startswith('state')),
                      init_loc_fn=init_to_sample)
    svi = SVI(torus_dbn, guide, Adam(dict(lr=0.1)), TraceEnum_ELBO())
    plot_rama(lengths, phis, psis, filename='ground_truth')
    total_iters = 100
    num_states = 55
    plot_rate = 5
    dataset = TensorDataset(phis, psis, lengths)
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_losses = []
    with tqdm.trange(total_iters) as pbar:
        total_loss = float('inf')
        for i in pbar:
            losses = []
            num_batches = 0
            for j, (phis, psis, lengths) in enumerate(dataloader):
                loss = svi.step(phis, psis, lengths, num_states=num_states)
                losses.append(loss)
                num_batches += 1
                pbar.set_description_str(f"SVI (batch {j}/{len(dataset)//batch_size}):"
                                         f" {loss / batch_size:.4} [epoch loss: {total_loss:.4}]",
                                         refresh=True)
            total_loss = np.sum(losses) / (batch_size * num_batches)
            total_losses.append(total_loss)
            pbar.set_description_str(f"SVI (batch {j}/{len(dataset)//batch_size}):"
                                     f" {loss / batch_size:.4} [epoch loss: {total_loss:.4}]",
                                     refresh=True)
            if i % plot_rate == 0:
                sample_and_plot(torus_dbn, guide, filename=f'learned_{i}',
                                num_sequences=len(dataset), num_states=num_states)
    sample_and_plot(torus_dbn, guide, filename=f'learned_finish',
                    num_sequences=len(dataset), num_states=num_states)
    plot_losses(total_losses)


if __name__ == '__main__':
    main(sys.argv)
