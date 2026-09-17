"""`random.simplex` must be uniform over the simplex, which docs/simplex.typ proves.

Uniformity is awkward to test directly because equal-volume bins on a simplex
are awkward. The proof's stick-breaking map turns a uniform simplex draw into an
iid uniform sample on the unit cube, where a regular grid does have equal-volume
bins, so a chi-square there is exact rather than asymptotic in the binning.

A single chi-square only ever fails to reject. The test that has teeth is that
the p-values of many independent trials are themselves uniform: a sampler with a
subtle bias drives them towards zero while leaving any one of them innocent.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pytest import approx
from scipy import stats

from raytrax.random import cosine_hemisphere, fixed_sorted, simplex, sphere

BINS = 6


@pytest.mark.parametrize("length", [1, 2, 3, 4])
def test_fixed_sorted_is_a_sort(length):
    samples = jax.random.uniform(jax.random.key(0), (100, length))
    assert np.array_equal(np.asarray(fixed_sorted(samples)), np.asarray(jnp.sort(samples, axis=-1)))


@functools.cache
def sampler(ncorner):
    """Compiled once per corner count; a fresh lambda per call would recompile."""
    return jax.jit(jax.vmap(lambda key: simplex(key, ncorner - 1)))


def weights(ncorner, count, seed):
    return np.asarray(sampler(ncorner)(jax.random.split(jax.random.key(seed), count)))


def to_cube(w):
    """Stick-breaking ratios pushed through their own Beta(1, k) CDFs."""
    ncorner = w.shape[-1]
    remaining = np.ones(w.shape[0])
    cube = np.empty((w.shape[0], ncorner - 1))
    for k in range(ncorner - 1):
        ratio = np.divide(w[:, k], remaining, out=np.zeros_like(remaining), where=remaining > 0)
        cube[:, k] = 1 - (1 - ratio) ** (ncorner - 1 - k)
        remaining = remaining - w[:, k]
    return cube


def uniformity_pvalue(w):
    cube = to_cube(w)
    counts, _ = np.histogramdd(cube, bins=[np.linspace(0, 1, BINS + 1)] * cube.shape[1])
    return stats.chisquare(counts.ravel(), np.full(counts.size, w.shape[0] / counts.size)).pvalue


@pytest.mark.parametrize("ncorner", [3, 4])
class TestSimplex:
    def test_weights_are_barycentric(self, ncorner):
        """Non-negative and summing to one, or the sampled point leaves the cell."""
        w = weights(ncorner, 100_000, 1)
        assert np.all(w >= 0)
        assert w.sum(axis=1) == approx(1.0, abs=1e-12)

    def test_density_is_constant(self, ncorner):
        """Equal-volume bins hold equal counts, in the transformed coordinates."""
        assert uniformity_pvalue(weights(ncorner, 200_000, 2)) > 0.001

    def test_bias_would_show_up_as_non_uniform_pvalues(self, ncorner):
        """The trial p-values are uniform, which is what a true null looks like."""
        p = np.array([uniformity_pvalue(weights(ncorner, 20_000, 5000 + i)) for i in range(64)])
        assert stats.kstest(p, "uniform").pvalue > 0.001
        assert np.mean(p < 0.05) < 0.15

    def test_a_skewed_sampler_is_caught(self, ncorner):
        """Sensitivity: a 5% tilt towards one corner must fail, so the check is not vacuous.

        Scaling one weight by 1.05 and renormalising is a bijection of the simplex,
        so the draws stay valid barycentric coordinates and only the density moves:
        by about 16% peak to trough on a triangle, 21% on a tetrahedron. Half that
        tilt is below what 200,000 draws resolve once the bins are spread over a
        tetrahedron, which is why the margin here is not finer.
        """
        w = weights(ncorner, 200_000, 3)
        tilted = w * (1 + 0.05 * np.eye(ncorner)[0])
        assert uniformity_pvalue(tilted / tilted.sum(axis=1, keepdims=True)) < 0.001


def test_sphere_is_isotropic():
    """Unit length, and no preferred direction."""
    keys = jax.random.split(jax.random.key(4), 200_000)
    points = np.asarray(jax.jit(jax.vmap(lambda key: sphere(key, 3)))(keys))
    assert np.linalg.norm(points, axis=1) == approx(1.0, abs=1e-12)
    assert points.mean(axis=0) == approx(np.zeros(3), abs=0.01)


def test_cosine_hemisphere_is_lambertian():
    """Unit length, in the normal's half-space, with mean cosine 2/3."""
    normal = jnp.array([0.0, 0.0, 1.0])
    keys = jax.random.split(jax.random.key(5), 200_000)
    directions = np.asarray(jax.jit(jax.vmap(lambda key: cosine_hemisphere(key, normal)))(keys))
    assert np.linalg.norm(directions, axis=1) == approx(1.0, abs=1e-12)
    assert np.all(directions @ np.asarray(normal) >= 0)
    # E[cos] under a cosine-weighted hemisphere is 2/3.
    assert (directions @ np.asarray(normal)).mean() == approx(2 / 3, abs=0.01)
