"""The march is only sound if a ray that leaves a cell arrives in the one the
topology says it should."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from raytrax import sources, transport
from raytrax.interactions import Kind
from raytrax.mesh import boundaries, geometry, topology
from raytrax.paths import AxisymmetricPath, LinearPath
from raytrax.scene import build as build_scene
from raytrax.scene import contains
from tests.meshes import brick, make, mixed_3d, uniform_boundary

# Jitted so that a vmap over many rays compiles once rather than dispatching op by op.
sample = eqx.filter_jit(jax.vmap(sources.sample, in_axes=(0, None, None, None)))
step = eqx.filter_jit(jax.vmap(transport.step, in_axes=(0, None, None, None, None, 0, 0, 0)))


def step_once(scene, mesh_geometry, path, seed=0, num=4000):
    """One step from volume-sampled rays: the cells they started in, the cells
    they ended in, and where they ended."""
    keys = jax.random.split(jax.random.key(seed), num)
    state, cells = sample(keys, sources.build(mesh_geometry.cells.volume), scene, path)
    table = uniform_boundary(Kind.ABSORB)
    nxt, moved, *_ = step(keys, scene, path, table, jnp.zeros(scene.num_cells), cells, state, jnp.ones(num))
    return cells, nxt, moved.position


@pytest.mark.parametrize("case", ["mixed_3d", "brick_3d", "brick_2d"])
def test_linear_rays_land_inside_their_next_cell(case):
    blocks, vertices = {"mixed_3d": mixed_3d(), "brick_3d": brick((3, 3, 3)), "brick_2d": brick((4, 4), dim=2)}[case]
    _, mesh_geometry, scene = make(blocks, vertices)
    cells, nxt, position = step_once(scene, mesh_geometry, LinearPath())

    # Interior crossings only; a boundary hit leaves the ray on the surface.
    interior = nxt != cells
    assert bool(jnp.any(interior))
    scale = float(np.cbrt(mesh_geometry.cells.volume.max()))
    assert bool(jnp.all(contains(scene, nxt[interior], position[interior], tolerance=1e-5 * scale)))


def test_axisymmetric_rays_land_inside_their_next_cell():
    """The hyperbolic crossing, on a revolved brick with cells against the axis."""
    blocks, vertices = brick((4, 4), dim=2)
    mesh_topology = topology.build(blocks)
    planar = geometry.build(mesh_topology, vertices)
    mesh_geometry = geometry.revolve(planar)
    on_axis = np.flatnonzero(mesh_topology.is_boundary & np.isclose(planar.faces.centroid[:, 1], 0.0))
    scene = build_scene(mesh_topology, mesh_geometry, boundaries.tag(mesh_topology, {"axis": on_axis}))
    cells, nxt, position = step_once(scene, mesh_geometry, AxisymmetricPath(), seed=2)

    interior = nxt != cells
    assert bool(jnp.any(interior))
    assert bool(jnp.all(contains(scene, nxt[interior], position[interior], tolerance=1e-9)))
