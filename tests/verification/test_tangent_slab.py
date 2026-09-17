import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import expn

from raytrax import interactions, sources, transport
from raytrax.interactions import Kind
from raytrax.mesh import boundaries, geometry, topology
from raytrax.paths import LinearPath
from raytrax.scene import build as build_scene
from tests.meshes import brick, brick_sides


@pytest.mark.parametrize("dim", [3, 2])
def test_tangent_slab(dim):
    """A hot wall radiating into cold absorbing gas has a known 1D profile.

    Mirroring the lateral sides makes the slab effectively infinite, so the
    cell-averaged heating must match 2*[E3(k*a) - E3(k*b)] / (b - a).

    The 2D mesh is read as a domain extruded out of plane, which is the same
    slab: it has to land on the same profile as the 3D one. That is the check
    that the out-of-plane component really is carried in |tangent|, since
    dropping it would shorten every path and pile the heating up at the wall.
    """
    depth, kappa, num_rays = 20, 0.25, 200_000
    blocks, vertices = brick((depth,) + (1,) * (dim - 1), dim=dim)

    mesh_topology = topology.build(blocks)
    mesh_geometry = geometry.build(mesh_topology, vertices)
    groups = boundaries.tag(mesh_topology, brick_sides(mesh_topology, mesh_geometry, vertices))
    scene = build_scene(mesh_topology, mesh_geometry, groups)

    lateral = ["ylo", "yhi", "zlo", "zhi"][: 2 * (dim - 1)]
    kinds = {"xlo": Kind.ABSORB, "xhi": Kind.ABSORB} | dict.fromkeys(lateral, Kind.SPECULAR)
    table = interactions.build(groups, kinds)

    hot = np.flatnonzero(groups.face_group == groups.index("xlo"))
    source = sources.build(jnp.zeros(mesh_topology.num_cells), mesh_geometry.faces.area[hot], hot)

    key_rays, key_march = jax.random.split(jax.random.key(0))
    state, cell = jax.vmap(sources.sample, in_axes=(0, None, None, None))(
        jax.random.split(key_rays, num_rays), source, scene, LinearPath()
    )
    rays = transport.Rays(
        cell=cell,
        state=state,
        energy=jnp.full(num_rays, source.total / num_rays),
        alive=jnp.ones(num_rays, dtype=bool),
    )
    _, deposit = transport.march(
        key_march, rays, scene, LinearPath(), table, jnp.full(mesh_topology.num_cells, kappa), max_steps=4000
    )

    order = np.argsort(mesh_geometry.cells.centroid[:, 0])
    heating = np.asarray(deposit.cells)[order] / mesh_geometry.cells.volume[order]

    near, far = np.arange(depth), np.arange(depth) + 1
    exact = 2 * (expn(3, kappa * near) - expn(3, kappa * far))

    # Only where the Monte-Carlo still has enough samples to be meaningful.
    resolved = slice(0, 12)
    assert np.abs(heating[resolved] / exact[resolved] - 1).max() < 0.02
