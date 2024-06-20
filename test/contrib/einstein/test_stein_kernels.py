# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
from copy import copy

import numpy as np
from numpy.testing import assert_allclose
import pytest

from jax import numpy as jnp, random

from numpyro import sample
from numpyro.distributions import Normal
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.initialization import init_to_value

from numpyro.contrib.einstein import SteinVI
from numpyro.contrib.einstein.stein_kernels import (
    GraphicalKernel,
    IMQKernel,
    LinearKernel,
    MixtureKernel,
    RandomFeatureKernel,
    RBFKernel,
    ProbabilityProductKernel
)
from numpyro.optim import Adam

T = namedtuple("TestSteinKernel", ["kernel", "particle_info", "loss_fn", "kval"])

PARTICLES = (np.array([[1.0, 2.0], [10.0, 5.0], [7.0, 3.0], [2.0, -1]]),)


def MOCK_MODEL(): sample('x', Normal())

TEST_CASES = [
    T(
        RBFKernel,
        lambda d: {},
        lambda x: x,
        {
            # let 
            #   median trick (x_i in PARTICLES)
            #   h = med( [||x_i-x_j||_2]_{i,j=(0,0)}^{(m,m)} )^2 / log(m) = 16.92703711264772
            #   x = (1, 2); y=(10,5)
            # in
            # k(x,y) = exp(-.5 * ||x-y||_2^2 / h) = 0.00490776
            "norm": 0.00490776,
            # let
            #   h = 16.92703711264772 (from norm case)
            #   x = (1, 2); y=(10,5)
            # in
            # k(x,y) = exp(-.5 * (x-y)^2 / h) = (0.00835209, 0.5876088)
            "vector": np.array([0.00835209, 0.5876088]),
            # I(n) is n by n identity matrix
            # let
            #   k_norm = 0.00490776  (from norm case)
            #   x = (1, 2); y=(10,5)
            # in 
            # k(x,y) = k_norm * I
            "matrix": np.array([[0.00490776, 0.0], [0.0, 0.00490776]]),
        },
    ),
    T(RandomFeatureKernel, lambda d: {}, lambda x: x, {"norm": 13.805723}),
    T(
        IMQKernel,
        lambda d: {},
        lambda x: x,
        {"norm": 0.104828484, "vector": np.array([0.11043153, 0.31622776])},
    ),
    T(LinearKernel, lambda d: {}, lambda x: x, {"norm": 21.0}),
    T(  # TODO: Add test cases for mixture with PPK, RBFHessian and CF
        lambda mode: MixtureKernel(
            mode=mode,
            ws=np.array([0.2, 0.8]),
            kernel_fns=[RBFKernel(mode), RBFKernel(mode)],
        ),
        lambda d: {},
        lambda x: x,
        # simply .2rbf_matrix + .8 rbf_matrix = rbf_matrix
        {"matrix": np.array([[0.00490776, 0.0], [0.0, 0.00490776]])},
    ),
    T(
        lambda mode: GraphicalKernel(
            mode=mode, local_kernel_fns={"p1": RBFKernel("norm")}
        ),
        lambda d: {"p1": (0, d)},
        lambda x: x,
        {"matrix": np.array([[0.00490776, 0.0], [0.0, 0.00490776]])},
    ),
    T(
        lambda mode: ProbabilityProductKernel(
            mode=mode, 
            guide=AutoNormal(MOCK_MODEL)
        ),
        lambda d: {'x_auto_loc': (0,1), 'x_auto_scale': (1,2)},
        lambda x: x,
        # eq. 5 Probability Product Kernels
        # x := (loc_x, softplus-inv(std_x)); y =: (loc_y, softplus-inv(std_y))
        # let s+(z) = log(exp(z)+1); x =(1,2); y=(10,5) in
        # k(x,y) = exp(-.5((1/s+(2))^2 + 
        #                  (10/s+(5))^2 - 
        #                  (1/(s+(2)^2 + (10/s+(5))^2)) ** 2 / (1/s+(2)^2 + 1/s+(5)^2)))
        #        = 0.2544481
        {"norm": 0.2544481},  
    ),
    # TODO:
    # T(
    #     lambda mode: CanonicalFunctionKernel(
    #     ),
    #     lambda d: {'x_auto_loc': (0,1), 'x_auto_scale': (1,2)},
    #     lambda x: x,
    #     {"norm": ...},  
    # ),
    # T(
    #     lambda mode: RBFHessianKernel(
    #         mode=mode, 
    #     ),
    #     lambda d: {'x_auto_loc': (0,1), 'x_auto_scale': (1,2)},
    #     lambda x: x,
    #     {"norm": ...},  
    # ),
]


TEST_IDS = [t[0].__class__.__name__ for t in TEST_CASES]


@pytest.mark.parametrize(
    "kernel, particle_info, loss_fn, kval", TEST_CASES, ids=TEST_IDS
)
@pytest.mark.parametrize("particles", PARTICLES)
@pytest.mark.parametrize("mode", ["norm", "vector", "matrix"])
def test_kernel_forward(
    kernel, particles, particle_info, loss_fn, mode, kval
):
    if mode not in kval:
        pytest.skip()
    (d,) = particles[0].shape
    kernel = kernel(mode=mode)
    kernel.init(random.PRNGKey(0), particles.shape)
    kernel_fn = kernel.compute(random.PRNGKey(0), particles, particle_info(d), loss_fn)
    value = kernel_fn(particles[0], particles[1])
    assert_allclose(value, jnp.array(kval[mode]), atol=1e-6)


@pytest.mark.parametrize(
    "kernel, particle_info, loss_fn, kval", TEST_CASES, ids=TEST_IDS
)
@pytest.mark.parametrize("mode", ["norm", "vector", "matrix"])
@pytest.mark.parametrize("particles", PARTICLES)
def test_apply_kernel(
    kernel, particles, particle_info, loss_fn, mode, kval
):
    if mode not in kval:
        pytest.skip()
    (d,) = particles[0].shape
    kernel_fn = kernel(mode=mode)
    kernel_fn.init(random.PRNGKey(0), particles.shape)
    kernel_fn = kernel_fn.compute(random.PRNGKey(0), particles, particle_info(d), loss_fn)
    v = np.ones_like(kval[mode])
    stein = SteinVI(id, id, Adam(1.0), kernel(mode))
    value = stein._apply_kernel(kernel_fn, particles[0], particles[1], v)
    kval_ = copy(kval)
    if mode == "matrix":
        kval_[mode] = np.dot(kval_[mode], v)
    assert_allclose(value, kval_[mode], atol=1e-6)
