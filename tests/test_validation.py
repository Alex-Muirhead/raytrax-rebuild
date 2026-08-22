import jax
import jax.numpy as jnp
import numpy as np
from meshes import brick, brick_sides
from scipy.special import expn

from raytrax import interactions, sources, transport
from raytrax.interactions import Kind
from raytrax.mesh import boundaries, geometry, topology
from raytrax.paths import LinearPath
from raytrax.scene import build as build_scene


def test_tangent_slab():
    """A hot wall radiating into cold absorbing gas has a known 1D profile.

    Mirroring the four lateral sides makes the slab effectively infinite, so the
    cell-averaged heating must match 2*[E3(k*a) - E3(k*b)] / (b - a).
    """
    depth, kappa, num_rays = 20, 0.25, 200_000
    blocks, vertices = brick((depth, 1, 1))

    mesh_topology = topology.build(blocks)
    mesh_geometry = geometry.build(mesh_topology, vertices)
    groups = boundaries.tag(mesh_topology, brick_sides(mesh_topology, mesh_geometry, vertices))
    scene = build_scene(mesh_topology, mesh_geometry, groups)

    table = interactions.build(
        groups,
        {
            "xlo": Kind.ABSORB,
            "xhi": Kind.ABSORB,
            "ylo": Kind.SPECULAR,
            "yhi": Kind.SPECULAR,
            "zlo": Kind.SPECULAR,
            "zhi": Kind.SPECULAR,
        },
    )

    hot = np.flatnonzero(groups.face_group == groups.index("xlo"))
    source = sources.Sources(
        cell_power=jnp.zeros(mesh_topology.num_cells),
        face_power=jnp.asarray(mesh_geometry.faces.area[hot]),
        face=jnp.asarray(hot),
    )

    key_rays, key_march = jax.random.split(jax.random.key(0))
    state, cell = jax.vmap(sources.sample, in_axes=(0, None, None))(jax.random.split(key_rays, num_rays), source, scene)
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
