import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import expn

from raytrax import interactions, sources, transport
from raytrax.interactions import Kind
from raytrax.mesh import boundaries, geometry, topology
from raytrax.paths import AxisymmetricPath, LinearPath
from raytrax.scene import build as build_scene
from tests.meshes import brick, brick_sides

sample = eqx.filter_jit(jax.vmap(sources.sample, in_axes=(0, None, None, None)))


@pytest.mark.parametrize("mode", ["3d", "2d", "axisymmetric"])
def test_tangent_slab(mode):
    """A hot wall radiating into cold absorbing gas has a known 1D profile.

    Mirroring the lateral sides makes the slab effectively infinite, so the
    cell-averaged heating must match 2*[E3(k*a) - E3(k*b)] / (b - a).

    The 2D mesh is read as a domain extruded out of plane, which is the same
    slab: it has to land on the same profile as the 3D one. That is the check
    that the out-of-plane component really is carried in |tangent|, since
    dropping it would shorten every path and pile the heating up at the wall.

    Revolved, the 2D mesh is a cylinder with a mirrored outer wall and the hot
    wall as one end. The wall's normal is radial, so a bounce leaves the axial
    direction cosine alone and the distance flown to reach a given depth is the
    slab's, so every ring must carry the same profile. Skipping `revolve` misses
    it by about 40% in each of the two rings.
    """
    depth, kappa, num_rays = 20, 0.25, 200_000
    dim = 3 if mode == "3d" else 2
    path = AxisymmetricPath() if mode == "axisymmetric" else LinearPath()
    # Two rings, because with one the ring weights scale the source and every
    # volume alike and cancel out of the heating.
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

    hot = np.flatnonzero(groups.face_group == groups.index("xlo"))
    source = sources.build(jnp.zeros(mesh_topology.num_cells), mesh_geometry.faces.area[hot], hot)

    key_rays, key_march = jax.random.split(jax.random.key(0))
    state, cell = sample(jax.random.split(key_rays, num_rays), source, scene, path)
    rays = transport.Rays(
        cell=cell,
        state=state,
        energy=jnp.full(num_rays, source.total / num_rays),
        alive=jnp.ones(num_rays, dtype=bool),
    )
    _, deposit = transport.march(
        key_march, rays, scene, path, table, jnp.full(mesh_topology.num_cells, kappa), max_steps=4000
    )

    # Rows of depth, one column per cell across the slab.
    order = np.lexsort(mesh_geometry.cells.centroid.T[::-1])
    heating = (np.asarray(deposit.cells)[order] / mesh_geometry.cells.volume[order]).reshape(depth, -1)

    near, far = np.arange(depth), np.arange(depth) + 1
    exact = 2 * (expn(3, kappa * near) - expn(3, kappa * far))

    # Only where the Monte-Carlo still has enough samples to be meaningful.
    resolved = slice(0, 12)
    assert np.abs(heating[resolved] / exact[resolved, None] - 1).max() < 0.02
