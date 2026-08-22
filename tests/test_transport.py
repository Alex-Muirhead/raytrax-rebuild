import jax
import jax.numpy as jnp
import numpy as np
import pytest
from meshes import brick, make, mixed_3d
from pytest import approx

from raytrax import interactions, sources, transport
from raytrax.interactions import Kind
from raytrax.paths import LinearPath
from raytrax.scene import contains


def emit_from_volume(scene, mesh_geometry):
    return sources.Sources(
        cell_power=jnp.asarray(mesh_geometry.cells.volume),
        face_power=jnp.zeros(0),
        face=jnp.zeros(0, dtype=int),
    )


def uniform_boundary(kind, reflectivity=1.0):
    """One implicit group covering every boundary face."""
    return interactions.Interactions(kind=jnp.array([kind]), reflectivity=jnp.array([reflectivity]))


def run(scene, source, kinds_table, seed=0, num_rays=20_000, kappa=1.0, max_steps=400):
    rays_key, march_key = jax.random.split(jax.random.key(seed))
    state, cell = jax.vmap(sources.sample, in_axes=(0, None, None))(jax.random.split(rays_key, num_rays), source, scene)
    rays = transport.Rays(
        cell=cell,
        state=state,
        energy=jnp.full(num_rays, source.total / num_rays),
        alive=jnp.ones(num_rays, dtype=bool),
    )
    absorptivity = jnp.full(scene.num_cells, kappa)
    return transport.march(march_key, rays, scene, LinearPath(), kinds_table, absorptivity, max_steps=max_steps)


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
        points = jax.vmap(sources.sample_cell_point, in_axes=(0, None, 0))(keys, scene, cells)
        assert bool(jnp.all(contains(scene, cells, points)))

    def test_cell_points_are_uniform(self):
        """Cells are picked in proportion to volume, so the mean must be the
        centroid of the whole brick, not of the cell list."""
        blocks, vertices = brick((2, 2, 2))
        _, mesh_geometry, scene = make(blocks, vertices)
        source = emit_from_volume(scene, mesh_geometry)
        state, _ = jax.vmap(sources.sample, in_axes=(0, None, None))(
            jax.random.split(jax.random.key(0), 100_000), source, scene
        )
        assert np.asarray(state.position.mean(axis=0)) == approx([1.0, 1.0, 1.0], abs=0.02)


class TestEnergyConservation:
    """The strongest invariant: in a sealed box nothing may go missing."""

    @pytest.mark.parametrize("case", ["brick_3d", "brick_2d", "mixed_3d"])
    def test_sealed_box_conserves_energy(self, case):
        blocks, vertices = {"brick_3d": brick((3, 3, 3)), "brick_2d": brick((4, 4), dim=2), "mixed_3d": mixed_3d()}[
            case
        ]
        _, mesh_geometry, scene = make(blocks, vertices)
        table = uniform_boundary(Kind.ABSORB)
        rays, deposit = run(scene, emit_from_volume(scene, mesh_geometry), table)

        emitted = float(mesh_geometry.cells.volume.sum())
        assert float(deposit.cells.sum() + deposit.faces.sum()) == approx(emitted, rel=1e-4)
        assert not bool(jnp.any(rays.alive)), "rays ran out of steps"

    def test_escaping_energy_is_not_deposited(self):
        blocks, vertices = brick((3, 3, 3))
        _, mesh_geometry, scene = make(blocks, vertices)
        table = uniform_boundary(Kind.ABSORB)
        escaping = uniform_boundary(Kind.ESCAPE)

        _, sealed = run(scene, emit_from_volume(scene, mesh_geometry), table)
        _, open_ = run(scene, emit_from_volume(scene, mesh_geometry), escaping)

        assert float(sealed.faces.sum()) > 0
        assert float(open_.faces.sum()) == approx(0.0)
        assert float(open_.cells.sum()) == approx(float(sealed.cells.sum()), rel=1e-6)


class TestReflection:
    def test_specular_reflection_keeps_rays_inside(self):
        """A fully mirrored box may only lose energy to absorption in the gas."""
        blocks, vertices = brick((3, 3, 3))
        _, mesh_geometry, scene = make(blocks, vertices)
        mirror = uniform_boundary(Kind.SPECULAR)
        _, deposit = run(scene, emit_from_volume(scene, mesh_geometry), mirror, max_steps=600, kappa=2.0)

        emitted = float(mesh_geometry.cells.volume.sum())
        assert float(deposit.faces.sum()) == approx(0.0)
        assert float(deposit.cells.sum()) == approx(emitted, rel=2e-3)


class TestCrossing:
    """The march is only sound if a ray that leaves a cell arrives in the one
    the topology says it should."""

    @pytest.mark.parametrize("case", ["mixed_3d", "brick_3d", "brick_2d"])
    def test_rays_land_inside_their_next_cell(self, case):
        blocks, vertices = {"mixed_3d": mixed_3d(), "brick_3d": brick((3, 3, 3)), "brick_2d": brick((4, 4), dim=2)}[
            case
        ]
        _, mesh_geometry, scene = make(blocks, vertices)
        table = uniform_boundary(Kind.ABSORB)

        num = 4000
        keys = jax.random.split(jax.random.key(0), num)
        state, cells = jax.vmap(sources.sample, in_axes=(0, None, None))(
            keys, emit_from_volume(scene, mesh_geometry), scene
        )

        nxt, moved, *_ = jax.vmap(transport.step, in_axes=(0, None, None, None, None, 0, 0, 0))(
            keys, scene, LinearPath(), table, jnp.zeros(scene.num_cells), cells, state, jnp.ones(num)
        )
        # Interior crossings only; a boundary hit leaves the ray on the surface.
        interior = nxt != cells
        assert bool(jnp.any(interior))
        scale = float(np.cbrt(mesh_geometry.cells.volume.max()))
        assert bool(jnp.all(contains(scene, nxt[interior], moved.position[interior], tolerance=1e-5 * scale)))


class TestDiffuseReflection:
    def test_diffuse_box_conserves_energy(self):
        """Resampling the direction at the wall must not create or destroy energy."""
        blocks, vertices = brick((3, 3, 3))
        _, mesh_geometry, scene = make(blocks, vertices)
        diffuse = uniform_boundary(Kind.DIFFUSE)
        _, deposit = run(scene, emit_from_volume(scene, mesh_geometry), diffuse, max_steps=600, kappa=2.0)

        emitted = float(mesh_geometry.cells.volume.sum())
        assert float(deposit.faces.sum()) == approx(0.0)
        assert float(deposit.cells.sum()) == approx(emitted, rel=2e-3)
