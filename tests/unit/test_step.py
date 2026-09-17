"""The march is only sound if a ray that leaves a cell arrives in the one the
topology says it should."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from raytrax import sources, transport
from raytrax.interactions import Kind
from raytrax.paths import AxisymmetricPath, LinearPath
from raytrax.scene import contains
from tests.meshes import SMALL_MESHES, brick, make, uniform_boundary

RAYS = 1000

# Jitted so that a vmap over many rays compiles once rather than dispatching op by op.
sample = eqx.filter_jit(jax.vmap(sources.sample, in_axes=(0, None, None, None)))
step = eqx.filter_jit(jax.vmap(transport.step, in_axes=(0, None, None, None, None, 0, 0, 0)))


def step_once(scene, mesh_geometry, path):
    """Rays sampled through the volume and stepped once, as `(start cell, end cell, end position)`."""
    keys = jax.random.split(jax.random.key(0), RAYS)
    state, cells = sample(keys, sources.build(mesh_geometry.cells.volume), scene, path)
    table = uniform_boundary(Kind.ABSORB)
    nxt, moved, *_ = step(keys, scene, path, table, jnp.zeros(scene.num_cells), cells, state, jnp.ones(RAYS))
    return cells, nxt, moved.position


@pytest.mark.parametrize("name", SMALL_MESHES)
def test_linear_rays_land_inside_their_next_cell(name):
    _, mesh_geometry, scene = make(*SMALL_MESHES[name]())
    cells, nxt, position = step_once(scene, mesh_geometry, LinearPath())

    # Interior crossings only; a boundary hit leaves the ray on the surface.
    interior = nxt != cells
    assert bool(jnp.any(interior))
    scale = float(np.cbrt(mesh_geometry.cells.volume.max()))
    assert bool(jnp.all(contains(scene, nxt[interior], position[interior], tolerance=1e-5 * scale)))


def test_axisymmetric_rays_land_inside_their_next_cell():
    """The hyperbolic crossing, on a revolved brick with cells against the axis."""
    _, mesh_geometry, scene = make(*brick((4, 4), dim=2), revolved=True)
    cells, nxt, position = step_once(scene, mesh_geometry, AxisymmetricPath())

    interior = nxt != cells
    assert bool(jnp.any(interior))
    assert bool(jnp.all(contains(scene, nxt[interior], position[interior], tolerance=1e-9)))
