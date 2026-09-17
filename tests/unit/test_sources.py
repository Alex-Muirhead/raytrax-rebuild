"""Where and in which direction a ray is launched."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pytest import approx

from raytrax import sources
from raytrax.paths import AxisymmetricPath, LinearPath
from raytrax.scene import contains
from tests.meshes import SMALL_MESHES, brick, make

DRAWS = 20_000

# Jitted so that a vmap over many draws compiles once rather than dispatching op by op.
sample = eqx.filter_jit(jax.vmap(sources.sample, in_axes=(0, None, None, None)))
sample_point = eqx.filter_jit(jax.vmap(sources.sample_point, in_axes=(0, None, None, 0, None)))


def keys(seed=0):
    return jax.random.split(jax.random.key(seed), DRAWS)


@pytest.mark.parametrize("name", SMALL_MESHES)
def test_cell_points_are_inside_their_cell(name):
    _, _, scene = make(*SMALL_MESHES[name]())
    cells = jax.random.randint(jax.random.key(1), (DRAWS,), 0, scene.num_cells)
    points = sample_point(keys(), scene.vertices, scene.cells.interior, cells, LinearPath())
    assert bool(jnp.all(contains(scene, cells, points)))


def test_cell_points_are_uniform():
    """Cells are picked in proportion to volume, so the mean is the centroid of
    the whole brick, not of the cell list."""
    _, mesh_geometry, scene = make(*brick((2, 2, 2)))
    state, _ = sample(keys(), sources.build(mesh_geometry.cells.volume), scene, LinearPath())
    assert np.asarray(state.position.mean(axis=0)) == approx([1.0, 1.0, 1.0], abs=0.02)


@pytest.mark.parametrize("dim,expected", [(3, 1.0), (2, np.pi / 4)])
def test_directions_keep_their_out_of_plane_share(dim, expected):
    """Directions are drawn in 3D and reduced to the mesh plane, so in 2D the
    stored tangent is the in-plane part: |tangent| = sin(theta) with cos(theta)
    uniform, whose mean is pi/4. Renormalising would give 1."""
    _, mesh_geometry, scene = make(*brick((3,) * dim, dim=dim))
    state, _ = sample(keys(), sources.build(mesh_geometry.cells.volume), scene, LinearPath())
    speed = np.linalg.norm(np.asarray(state.tangent), axis=-1)
    assert speed.max() <= 1 + 1e-6
    assert speed.mean() == approx(expected, abs=0.005)


def test_revolved_points_fill_the_ring_not_the_slice():
    """Uniform in the ring means the density carries a factor r. Over r in [1, 2]
    the mean radius is (integral r^2) / (integral r) = 14/9, where uniform in
    the slice would give 3/2."""
    blocks, vertices = [("quad", np.arange(4)[None, :])], np.array([[0, 1], [1, 1], [1, 2], [0, 2]], dtype=float)
    _, _, scene = make(blocks, vertices, revolved=True)
    cells = jnp.zeros(DRAWS, dtype=int)
    points = sample_point(keys(), scene.vertices, scene.cells.interior, cells, AxisymmetricPath())
    assert bool(jnp.all(contains(scene, cells, points)))
    assert float(points[:, 1].mean()) == approx(14 / 9, abs=0.01)
