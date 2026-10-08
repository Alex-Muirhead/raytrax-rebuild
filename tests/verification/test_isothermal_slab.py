import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import expn

from raytrax import interactions, sources, transport
from raytrax.interactions import Kind
from raytrax.mesh import boundaries, geometry, topology
from raytrax.paths import AxisymmetricPath, LinearPath
from raytrax.problem import Problem
from raytrax.scene import build as build_scene
from tests.meshes import brick, brick_sides


@pytest.mark.parametrize("mode", ["3d", "2d", "axisymmetric"])
def test_isothermal_slab(mode):
    """A hot gas between two cold black walls has a known 1D incident radiation.

    Each cell emits 4 * kappa * V for an emissive power of 1. With the lateral
    sides mirrored the slab is infinite, so the cell-averaged incident
    radiation must match 4 - 2 * [E3(k*a) - E3(k*b) + E3(k*(L-b)) - E3(k*(L-a))] / (k * (b - a)).

    This is the volume-emission counterpart of the tangent slab: the source is
    every cell rather than one wall, and the mirrors return what the gas emits
    sideways, so a wrong reflection shows up as a deficit across the whole
    depth rather than only far from the wall.
    """
    depth, kappa, num_rays = 20, 0.5, 1_000_000
    dim = 3 if mode == "3d" else 2
    path = AxisymmetricPath() if mode == "axisymmetric" else LinearPath()
    # Two rings, because with one the ring weights scale the source and every
    # volume alike and cancel out.
    across = 2 if mode == "axisymmetric" else 1
    blocks, vertices = brick((depth,) + (across,) * (dim - 1), dim=dim)

    mesh_topology = topology.build(blocks)
    mesh_geometry = geometry.build(mesh_topology, vertices)
    groups = boundaries.tag(mesh_topology, brick_sides(mesh_topology, mesh_geometry, vertices))
    if mode == "axisymmetric":
        mesh_geometry = geometry.revolve(mesh_geometry)
    scene = build_scene(mesh_topology, mesh_geometry, groups)

    # In the axisymmetric mode `ylo` is the axis, whose faces have no area.
    lateral = ["ylo", "yhi", "zlo", "zhi"][: 2 * (dim - 1)]
    kinds = {"xlo": Kind.ABSORB, "xhi": Kind.ABSORB} | dict.fromkeys(lateral, Kind.SPECULAR)
    table = interactions.build(groups, kinds)

    source = sources.build(4 * kappa * jnp.asarray(mesh_geometry.cells.volume))

    problem = Problem(scene, path, table, jnp.full(mesh_topology.num_cells, kappa), source)
    tally = transport.trace(jax.random.key(0), problem, num_rays=num_rays, lanes=num_rays, max_steps=4000)
    deposit = tally.estimate(problem)

    # Rows of depth, one column per cell across the slab.
    order = np.lexsort(mesh_geometry.cells.centroid.T[::-1])
    incident = (np.asarray(deposit.cells)[order] / (kappa * mesh_geometry.cells.volume[order])).reshape(depth, -1)

    near, far = kappa * np.arange(depth), kappa * (np.arange(depth) + 1)
    walls = expn(3, near) - expn(3, far) + expn(3, kappa * depth - far) - expn(3, kappa * depth - near)
    exact = 4 - 2 * walls / kappa

    assert np.abs(incident / exact[:, None] - 1).max() < 0.02
