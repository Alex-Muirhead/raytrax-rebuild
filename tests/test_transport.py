import jax
import jax.numpy as jnp
import numpy as np
import pytest
from meshes import brick, make, mixed_3d
from pytest import approx

from raytrax import interactions, sources, transport
from raytrax import scene as scene_module
from raytrax.interactions import Kind
from raytrax.mesh import boundaries, geometry, topology
from raytrax.paths import AxisymmetricPath, LinearPath
from raytrax.scene import contains


def emit_from_volume(scene, mesh_geometry):
    return sources.build(mesh_geometry.cells.volume)


def uniform_boundary(kind, reflectivity=1.0):
    """One implicit group covering every boundary face."""
    return interactions.Interactions(kind=(Kind(kind),), reflectivity=jnp.array([reflectivity]))


def run(scene, source, kinds_table, seed=0, num_rays=20_000, kappa=1.0, max_steps=400, path=None, **march_kwargs):
    path = path or LinearPath()
    rays_key, march_key = jax.random.split(jax.random.key(seed))
    state, cell = jax.vmap(sources.sample, in_axes=(0, None, None, None))(
        jax.random.split(rays_key, num_rays), source, scene, path
    )
    rays = transport.Rays(
        cell=cell,
        state=state,
        energy=jnp.full(num_rays, source.total / num_rays),
        alive=jnp.ones(num_rays, dtype=bool),
    )
    absorptivity = jnp.full(scene.num_cells, kappa)
    return transport.march(march_key, rays, scene, path, kinds_table, absorptivity, max_steps=max_steps, **march_kwargs)


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
        points = jax.vmap(sources.sample_point, in_axes=(0, None, None, 0, None))(
            keys, scene.vertices, scene.cells.interior, cells, LinearPath()
        )
        assert bool(jnp.all(contains(scene, cells, points)))

    @pytest.mark.parametrize("dim,expected", [(3, 1.0), (2, np.pi / 4)])
    def test_directions_keep_their_out_of_plane_share(self, dim, expected):
        """Directions are drawn in 3D and reduced to the mesh plane, so in 2D the
        stored tangent is the in-plane part: |tangent| = sin(theta) with
        cos(theta) uniform, whose mean is pi/4. Renormalising would give 1."""
        blocks, vertices = brick((3,) * dim, dim=dim)
        _, mesh_geometry, scene = make(blocks, vertices)
        state, _ = jax.vmap(sources.sample, in_axes=(0, None, None, None))(
            jax.random.split(jax.random.key(0), 100_000), emit_from_volume(scene, mesh_geometry), scene, LinearPath()
        )
        speed = jnp.linalg.vector_norm(state.tangent, axis=-1)
        assert float(speed.max()) <= 1 + 1e-6
        assert float(speed.mean()) == approx(expected, abs=3e-3)

    def test_cell_points_are_uniform(self):
        """Cells are picked in proportion to volume, so the mean must be the
        centroid of the whole brick, not of the cell list."""
        blocks, vertices = brick((2, 2, 2))
        _, mesh_geometry, scene = make(blocks, vertices)
        source = emit_from_volume(scene, mesh_geometry)
        state, _ = jax.vmap(sources.sample, in_axes=(0, None, None, None))(
            jax.random.split(jax.random.key(0), 100_000), source, scene, LinearPath()
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


class TestExtinction:
    """Stopping a faint ray is a bookkeeping move, not a physical one: it must
    move energy forward in time without moving it in space."""

    def mirrored_box(self):
        """A box a ray can never leave, so only extinction ever ends the march."""
        blocks, vertices = brick((3, 3, 3))
        _, mesh_geometry, scene = make(blocks, vertices)
        return mesh_geometry, scene, uniform_boundary(Kind.SPECULAR)

    def test_the_deposited_total_is_the_emitted_total(self):
        """The remainder is deposited rather than dropped, so the balance closes
        to rounding and not to the size of the floor."""
        mesh_geometry, scene, mirror = self.mirrored_box()
        source = emit_from_volume(scene, mesh_geometry)
        rays, deposit = run(scene, source, mirror, kappa=2.0, max_steps=2000, extinction=1e-6)

        assert not bool(jnp.any(rays.alive)), "extinction never ended the march"
        assert float(deposit.cells.sum() + deposit.faces.sum()) == approx(float(source.total), rel=1e-12)

    def test_a_coarser_floor_does_not_move_the_answer(self):
        """Two floors six decades apart must agree far more closely than the
        deposit varies across the mesh, or the truncation is displacing energy."""
        mesh_geometry, scene, mirror = self.mirrored_box()
        source = emit_from_volume(scene, mesh_geometry)
        loose, fine = run(scene, source, mirror, kappa=2.0, max_steps=4000, extinction=1e-12)
        _, coarse = run(scene, source, mirror, kappa=2.0, max_steps=4000, extinction=1e-6)

        assert not bool(jnp.any(loose.alive)), "the finer floor ran out of steps"
        drift = float(jnp.abs(coarse.cells - fine.cells).max() / fine.cells.max())
        assert drift < 1e-5

    def test_the_floor_is_what_ends_a_mirrored_march(self):
        """Without it the same box never terminates, which is the cost the floor
        buys back. The step cap is short of where `exp(-kappa s)` underflows,
        since reaching zero would end the march on its own."""
        mesh_geometry, scene, mirror = self.mirrored_box()
        source = emit_from_volume(scene, mesh_geometry)
        forever, _ = run(scene, source, mirror, kappa=1.0, max_steps=300, extinction=0.0)
        stopped, _ = run(scene, source, mirror, kappa=1.0, max_steps=300)

        assert bool(jnp.all(forever.alive))
        assert not bool(jnp.any(stopped.alive))


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
        state, cells = jax.vmap(sources.sample, in_axes=(0, None, None, None))(
            keys, emit_from_volume(scene, mesh_geometry), scene, LinearPath()
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
    @pytest.mark.parametrize("case", ["brick_3d", "brick_2d"])
    def test_diffuse_box_conserves_energy(self, case):
        """Resampling the direction at the wall must not create or destroy energy.

        In 2D the resampled direction is drawn about the wall normal embedded out
        of plane and then reduced, so this covers that path too."""
        blocks, vertices = {"brick_3d": brick((3, 3, 3)), "brick_2d": brick((4, 4), dim=2)}[case]
        _, mesh_geometry, scene = make(blocks, vertices)
        diffuse = uniform_boundary(Kind.DIFFUSE)
        _, deposit = run(scene, emit_from_volume(scene, mesh_geometry), diffuse, max_steps=600, kappa=2.0)

        emitted = float(mesh_geometry.cells.volume.sum())
        assert float(deposit.faces.sum()) == approx(0.0)
        assert float(deposit.cells.sum()) == approx(emitted, rel=2e-3)


class TestRefill:
    """Refilling dead lanes must change only the cost, never the answer."""

    @pytest.mark.parametrize("lanes,interval", [(20_000, 1), (5_000, 8), (2_000, 16)])
    def test_refilled_trace_launches_its_full_budget(self, lanes, interval):
        """In a sealed box every launched ray deposits its whole quantum, so the
        deposited total is exact evidence that the budget was spent exactly."""
        blocks, vertices = brick((3, 3, 3))
        _, mesh_geometry, scene = make(blocks, vertices)
        source = sources.build(mesh_geometry.cells.volume)

        _, deposit = transport.trace(
            jax.random.key(0),
            scene,
            LinearPath(),
            uniform_boundary(Kind.ABSORB),
            jnp.full(scene.num_cells, 1.0),
            source,
            num_rays=20_000,
            lanes=lanes,
            refill_interval=interval,
        )
        assert float(deposit.cells.sum() + deposit.faces.sum()) == approx(float(source.total), rel=1e-6)


class TestReducedState:
    """A 2D mesh is read as extruded out of plane. The component that points out
    of it is not stored -- it survives as the deficit in |tangent| -- so these
    guard the two things that makes true."""

    def launch(self, num, seed=0):
        blocks, vertices = brick((3, 3), dim=2)
        _, mesh_geometry, scene = make(blocks, vertices)
        state, cell = jax.vmap(sources.sample, in_axes=(0, None, None, None))(
            jax.random.split(jax.random.key(seed), num), emit_from_volume(scene, mesh_geometry), scene, LinearPath()
        )
        rays = transport.Rays(cell=cell, state=state, energy=jnp.ones(num), alive=jnp.ones(num, dtype=bool))
        return scene, rays

    def test_specular_march_preserves_the_out_of_plane_share(self):
        """Crossings and mirror bounces are the whole march, and neither may move
        energy between the plane and the direction out of it."""
        scene, rays = self.launch(2000)
        final, _ = transport.march(
            jax.random.key(1),
            rays,
            scene,
            LinearPath(),
            uniform_boundary(Kind.SPECULAR),
            jnp.full(scene.num_cells, 0.5),
            max_steps=200,
        )
        before = jnp.linalg.vector_norm(rays.state.tangent, axis=-1)
        after = jnp.linalg.vector_norm(final.state.tangent, axis=-1)
        assert np.allclose(np.asarray(before), np.asarray(after), rtol=1e-5)

    def test_travel_is_the_three_dimensional_length(self):
        """`travel` is the distance flown in 3D, which is the in-plane distance
        divided by |tangent|. Rays fly straight out, so the two are comparable."""
        scene, rays = self.launch(2000)
        final, _ = transport.march(
            jax.random.key(1),
            rays,
            scene,
            LinearPath(),
            uniform_boundary(Kind.ESCAPE),
            jnp.zeros(scene.num_cells),
            max_steps=100,
        )
        assert not bool(jnp.any(final.alive)), "rays ran out of steps"
        chord = jnp.linalg.vector_norm(final.state.position - rays.state.position, axis=-1)
        speed = jnp.linalg.vector_norm(rays.state.tangent, axis=-1)
        assert np.allclose(np.asarray(chord), np.asarray(final.state.travel * speed), rtol=1e-4)


class TestAxisymmetric:
    """A 2D mesh read as revolved. The geometry must be the ring's, and the
    march must not be able to leak through the symmetry axis."""

    def revolved(self, shape=(4, 4)):
        """A brick spanning the axis, weighted as the solid it sweeps."""
        blocks, vertices = brick(shape, dim=2)
        mesh_topology = topology.build(blocks)
        mesh_geometry = geometry.revolve(geometry.build(mesh_topology, vertices))
        on_axis = np.flatnonzero(
            mesh_topology.is_boundary & np.isclose(geometry.build(mesh_topology, vertices).faces.centroid[:, 1], 0.0)
        )
        tagged = boundaries.tag(mesh_topology, {"axis": on_axis})
        return mesh_geometry, tagged, scene_module.build(mesh_topology, mesh_geometry, tagged)

    def test_sampled_points_fill_the_ring_not_the_slice(self):
        """Uniform in the ring means the density carries a factor r, so the mean
        radius exceeds the meridional one. For a slab from a to b that is
        (integral r^2)/(integral r), which is 14/9 over [1, 2]."""
        vertices = np.array([[0, 1], [1, 1], [1, 2], [0, 2]], dtype=float)
        mesh_topology = topology.build([("quad", np.arange(4)[None, :])])
        mesh_geometry = geometry.revolve(geometry.build(mesh_topology, vertices))
        scene = scene_module.build(mesh_topology, mesh_geometry)

        keys = jax.random.split(jax.random.key(0), 200_000)
        cells = jnp.zeros(200_000, dtype=int)
        points = jax.vmap(sources.sample_point, in_axes=(0, None, None, 0, None))(
            keys, scene.vertices, scene.cells.interior, cells, AxisymmetricPath()
        )
        assert float(points[:, 1].mean()) == approx(14 / 9, abs=3e-3)
        assert bool(jnp.all(contains(scene, cells, points)))

    def test_sealed_box_conserves_energy(self):
        """The same invariant as the Cartesian modes, over ring volumes and with
        cells that sit against the axis."""
        mesh_geometry, _, scene = self.revolved()
        table = uniform_boundary(Kind.ABSORB)
        source = sources.build(mesh_geometry.cells.volume)
        rays, deposit = run(scene, source, table, path=AxisymmetricPath(), max_steps=600)

        assert float(deposit.cells.sum() + deposit.faces.sum()) == approx(float(source.total), rel=1e-4)
        assert not bool(jnp.any(rays.alive)), "rays ran out of steps"

    def test_nothing_escapes_through_the_axis(self):
        """A ray only reaches r = 0 with no swirl at all, which is measure zero.
        Any deposit on the axis group is a leak through the symmetry plane."""
        mesh_geometry, groups, scene = self.revolved()
        table = uniform_boundary(Kind.ABSORB)
        _, deposit = run(
            scene, sources.build(mesh_geometry.cells.volume), table, path=AxisymmetricPath(), max_steps=600
        )

        on_axis = np.flatnonzero(groups.face_group == groups.index("axis"))
        assert float(np.asarray(deposit.faces)[on_axis].sum()) == approx(0.0)
        assert float(deposit.faces.sum()) > 0, "the other walls must still absorb"

    def test_rays_land_inside_their_next_cell(self):
        """The hyperbolic march is only sound if a crossing arrives where the
        topology says it should."""
        mesh_geometry, _, scene = self.revolved()
        path = AxisymmetricPath()
        num = 4000
        keys = jax.random.split(jax.random.key(2), num)
        state, cells = jax.vmap(sources.sample, in_axes=(0, None, None, None))(
            keys, sources.build(mesh_geometry.cells.volume), scene, path
        )
        nxt, moved, *_ = jax.vmap(transport.step, in_axes=(0, None, None, None, None, 0, 0, 0))(
            keys, scene, path, uniform_boundary(Kind.ABSORB), jnp.zeros(scene.num_cells), cells, state, jnp.ones(num)
        )
        interior = nxt != cells
        assert bool(jnp.any(interior))
        assert bool(jnp.all(contains(scene, nxt[interior], moved.position[interior], tolerance=1e-9)))
