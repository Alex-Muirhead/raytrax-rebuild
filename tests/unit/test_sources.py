import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pytest import approx

from raytrax import scene as scene_module
from raytrax import sources
from raytrax.mesh import geometry, topology
from raytrax.paths import AxisymmetricPath, LinearPath
from raytrax.scene import contains
from tests.meshes import brick, make, mixed_3d

# Jitted so that a vmap over many draws compiles once rather than dispatching op by op.
sample = eqx.filter_jit(jax.vmap(sources.sample, in_axes=(0, None, None, None)))
sample_point = eqx.filter_jit(jax.vmap(sources.sample_point, in_axes=(0, None, None, 0, None)))


class TestSampling:
    """Sampled points must land inside the cell or on the face they came from."""

    @pytest.mark.parametrize("case", ["mixed_3d", "brick_3d", "brick_2d"])
    def test_cell_points_are_inside_their_cell(self, case):
        blocks, vertices = {"mixed_3d": mixed_3d(), "brick_3d": brick((2, 2, 2)), "brick_2d": brick((3, 3), dim=2)}[
            case
        ]
        _, _, scene = make(blocks, vertices)
        keys = jax.random.split(jax.random.key(0), 2000)
        cells = jax.random.randint(jax.random.key(1), (2000,), 0, scene.num_cells)
        points = sample_point(keys, scene.vertices, scene.cells.interior, cells, LinearPath())
        assert bool(jnp.all(contains(scene, cells, points)))

    @pytest.mark.parametrize("dim,expected", [(3, 1.0), (2, np.pi / 4)])
    def test_directions_keep_their_out_of_plane_share(self, dim, expected):
        """Directions are drawn in 3D and reduced to the mesh plane, so in 2D the
        stored tangent is the in-plane part: |tangent| = sin(theta) with
        cos(theta) uniform, whose mean is pi/4. Renormalising would give 1."""
        blocks, vertices = brick((3,) * dim, dim=dim)
        _, mesh_geometry, scene = make(blocks, vertices)
        state, _ = sample(
            jax.random.split(jax.random.key(0), 100_000), sources.build(mesh_geometry.cells.volume), scene, LinearPath()
        )
        speed = jnp.linalg.vector_norm(state.tangent, axis=-1)
        assert float(speed.max()) <= 1 + 1e-6
        assert float(speed.mean()) == approx(expected, abs=3e-3)

    def test_cell_points_are_uniform(self):
        """Cells are picked in proportion to volume, so the mean must be the
        centroid of the whole brick, not of the cell list."""
        blocks, vertices = brick((2, 2, 2))
        _, mesh_geometry, scene = make(blocks, vertices)
        state, _ = sample(
            jax.random.split(jax.random.key(0), 100_000), sources.build(mesh_geometry.cells.volume), scene, LinearPath()
        )
        assert np.asarray(state.position.mean(axis=0)) == approx([1.0, 1.0, 1.0], abs=0.02)

    def test_revolved_points_fill_the_ring_not_the_slice(self):
        """Uniform in the ring means the density carries a factor r, so the mean
        radius exceeds the unweighted one. For a slab from a to b that is
        (integral r^2)/(integral r), which is 14/9 over [1, 2]."""
        vertices = np.array([[0, 1], [1, 1], [1, 2], [0, 2]], dtype=float)
        mesh_topology = topology.build([("quad", np.arange(4)[None, :])])
        mesh_geometry = geometry.revolve(geometry.build(mesh_topology, vertices))
        scene = scene_module.build(mesh_topology, mesh_geometry)

        keys = jax.random.split(jax.random.key(0), 200_000)
        cells = jnp.zeros(200_000, dtype=int)
        points = sample_point(keys, scene.vertices, scene.cells.interior, cells, AxisymmetricPath())
        assert float(points[:, 1].mean()) == approx(14 / 9, abs=3e-3)
        assert bool(jnp.all(contains(scene, cells, points)))
