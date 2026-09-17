import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pytest import approx

from raytrax import scene as scene_module
from raytrax import sources, transport, tuning
from raytrax.interactions import Kind
from raytrax.mesh import boundaries, geometry, topology
from raytrax.paths import AxisymmetricPath, LinearPath
from raytrax.problem import Problem
from tests.meshes import brick, make, mixed_3d, uniform_boundary


def problem(scene, source, kinds_table, kappa=1.0, path=None):
    """A problem on a test mesh, with one absorption coefficient everywhere."""
    return Problem(scene, path or LinearPath(), kinds_table, jnp.full(scene.num_cells, kappa), source)


def run(scene, source, kinds_table, seed=0, num_rays=20_000, kappa=1.0, max_steps=400, path=None, **trace_kwargs):
    """Trace `num_rays` rays in as many lanes, as `(lanes, deposit)`."""
    case = problem(scene, source, kinds_table, kappa, path)
    tally = transport.trace(
        jax.random.key(seed), case, num_rays=num_rays, lanes=num_rays, max_steps=max_steps, **trace_kwargs
    )
    return tally.rays, tally.estimate(case)


class TestEnergyConservation:
    """The strongest invariant: in a sealed box nothing may go missing."""

    @pytest.mark.parametrize("case", ["brick_3d", "brick_2d", "mixed_3d"])
    def test_sealed_box_conserves_energy(self, case):
        blocks, vertices = {"brick_3d": brick((3, 3, 3)), "brick_2d": brick((4, 4), dim=2), "mixed_3d": mixed_3d()}[
            case
        ]
        _, mesh_geometry, scene = make(blocks, vertices)
        table = uniform_boundary(Kind.ABSORB)
        rays, deposit = run(scene, sources.build(mesh_geometry.cells.volume), table)

        emitted = float(mesh_geometry.cells.volume.sum())
        assert float(deposit.cells.sum() + deposit.faces.sum()) == approx(emitted, rel=1e-4)
        assert not bool(jnp.any(rays.alive)), "rays ran out of steps"

    def test_escaping_energy_is_not_deposited(self):
        blocks, vertices = brick((3, 3, 3))
        _, mesh_geometry, scene = make(blocks, vertices)
        table = uniform_boundary(Kind.ABSORB)
        escaping = uniform_boundary(Kind.ESCAPE)

        _, sealed = run(scene, sources.build(mesh_geometry.cells.volume), table)
        _, open_ = run(scene, sources.build(mesh_geometry.cells.volume), escaping)

        assert float(sealed.faces.sum()) > 0
        assert float(open_.faces.sum()) == approx(0.0)
        assert float(open_.cells.sum()) == approx(float(sealed.cells.sum()), rel=1e-6)
        # Same rays reaching the same faces, so what one box absorbs there is
        # exactly what the other lets out.
        assert float(open_.escaped) == approx(float(sealed.faces.sum()), rel=1e-12)


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
        source = sources.build(mesh_geometry.cells.volume)
        rays, deposit = run(scene, source, mirror, kappa=2.0, max_steps=2000, extinction=1e-6)

        assert not bool(jnp.any(rays.alive)), "extinction never ended the march"
        assert float(deposit.cells.sum() + deposit.faces.sum()) == approx(float(source.total), rel=1e-12)

    def test_a_coarser_floor_does_not_move_the_answer(self):
        """Two floors six decades apart must agree far more closely than the
        deposit varies across the mesh, or the truncation is displacing energy."""
        mesh_geometry, scene, mirror = self.mirrored_box()
        source = sources.build(mesh_geometry.cells.volume)
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
        source = sources.build(mesh_geometry.cells.volume)
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
        _, deposit = run(scene, sources.build(mesh_geometry.cells.volume), mirror, max_steps=600, kappa=2.0)

        emitted = float(mesh_geometry.cells.volume.sum())
        assert float(deposit.faces.sum()) == approx(0.0)
        assert float(deposit.cells.sum()) == approx(emitted, rel=2e-3)


class TestDiffuseReflection:
    @pytest.mark.parametrize("case", ["brick_3d", "brick_2d"])
    def test_diffuse_box_conserves_energy(self, case):
        """Resampling the direction at the wall must not create or destroy energy.

        In 2D the resampled direction is drawn about the wall normal embedded out
        of plane and then reduced, so this covers that path too."""
        blocks, vertices = {"brick_3d": brick((3, 3, 3)), "brick_2d": brick((4, 4), dim=2)}[case]
        _, mesh_geometry, scene = make(blocks, vertices)
        diffuse = uniform_boundary(Kind.DIFFUSE)
        _, deposit = run(scene, sources.build(mesh_geometry.cells.volume), diffuse, max_steps=600, kappa=2.0)

        emitted = float(mesh_geometry.cells.volume.sum())
        assert float(deposit.faces.sum()) == approx(0.0)
        assert float(deposit.cells.sum()) == approx(emitted, rel=2e-3)


class TestRefill:
    """Refilling dead lanes must change only the cost, never the answer."""

    @pytest.mark.parametrize("lanes,interval", [(20_000, 1), (5_000, 8), (2_000, 16)])
    def test_refilled_trace_launches_its_full_budget(self, lanes, interval):
        """In a sealed box every launched ray deposits all of its energy, so the
        deposited total is exact evidence that the budget was spent exactly."""
        blocks, vertices = brick((3, 3, 3))
        _, mesh_geometry, scene = make(blocks, vertices)
        source = sources.build(mesh_geometry.cells.volume)
        case = problem(scene, source, uniform_boundary(Kind.ABSORB))

        tally = transport.trace(jax.random.key(0), case, num_rays=20_000, lanes=lanes, refill_interval=interval)
        deposit = tally.estimate(case)
        assert int(tally.launched) == 20_000
        assert float(deposit.cells.sum() + deposit.faces.sum()) == approx(float(source.total), rel=1e-6)

    def test_what_escaped_survives_the_lane_being_reused(self):
        """An escaping ray deposits nothing anywhere, so the only record of it is
        the `escaped` total; the lane it dies in is handed straight to the next
        ray and would take that record with it."""
        blocks, vertices = brick((3, 3, 3))
        _, mesh_geometry, scene = make(blocks, vertices)
        source = sources.build(mesh_geometry.cells.volume)
        case = problem(scene, source, uniform_boundary(Kind.ESCAPE))

        tally = transport.trace(jax.random.key(0), case, num_rays=20_000, lanes=2_000)
        deposit = tally.estimate(case)
        assert float(deposit.escaped) > 0, "nothing escaped, so the budget closing proves nothing"
        accounted = float(deposit.cells.sum() + deposit.faces.sum() + deposit.escaped + tally.in_flight(case))
        assert accounted == approx(float(source.total), rel=1e-6)


class TestResume:
    """A tally extended by a second run must be the tally of one longer run."""

    def box(self, kind):
        blocks, vertices = brick((3, 3, 3))
        _, mesh_geometry, scene = make(blocks, vertices)
        return problem(scene, sources.build(mesh_geometry.cells.volume), uniform_boundary(kind))

    @staticmethod
    def accounted(tally, case):
        deposit = tally.estimate(case)
        return float(deposit.cells.sum() + deposit.faces.sum() + deposit.escaped + tally.in_flight(case))

    def test_a_resumed_tally_counts_both_runs(self):
        """The second run has a different ray count on the same lanes. Every ray
        of both runs must be launched once, and the balance must close over all of them."""
        case = self.box(Kind.ABSORB)
        tally = transport.trace(jax.random.key(0), case, num_rays=10_000, lanes=2_000)
        tally = transport.resume(tally, case, num_rays=5_000)

        deposit = tally.estimate(case)
        assert int(tally.launched) == 15_000
        assert tally.rays.cell.shape == (2_000,)
        assert float(deposit.cells.sum() + deposit.faces.sum()) == approx(float(case.sources.total), rel=1e-6)

    def test_rays_cut_off_by_the_step_limit_finish_on_resume(self):
        """In a mirrored box the rays outlive a short step limit. They must stay
        in the tally as energy in flight, and a resume with no new rays must
        finish them without losing any of it."""
        case = self.box(Kind.SPECULAR)
        tally = transport.trace(jax.random.key(0), case, num_rays=2_000, lanes=2_000, max_steps=16)
        assert bool(jnp.any(tally.rays.alive)), "the step limit did not cut any ray off"
        assert self.accounted(tally, case) == approx(float(case.sources.total), rel=1e-9)

        tally = transport.resume(tally, case, num_rays=0, max_steps=4_000)
        assert not bool(jnp.any(tally.rays.alive)), "the resume did not finish the rays"
        assert int(tally.launched) == 2_000
        assert self.accounted(tally, case) == approx(float(case.sources.total), rel=1e-9)

    def test_a_split_run_estimates_what_one_run_does(self):
        """Each ray escapes with a share in [0, 1] of the source power, so its
        variance is at most p (1 - p). Two independent estimates from 20_000 rays
        must then agree within 4 of their combined standard errors."""
        case = self.box(Kind.ESCAPE)
        single = transport.trace(jax.random.key(0), case, num_rays=20_000, lanes=4_096)
        split = transport.trace(jax.random.key(1), case, num_rays=10_000, lanes=4_096)
        split = transport.resume(split, case, num_rays=10_000)

        total = float(case.sources.total)
        share = float(single.estimate(case).escaped) / total
        assert 0 < share < 1
        error = total * np.sqrt(2 * share * (1 - share) / 20_000)
        assert abs(float(split.estimate(case).escaped) - share * total) < 4 * error


class TestLaneSizing:
    """How wide a batch `trace` marches, and the ceiling it must stay under.
    The ceiling reaches through what equinox and XLA return, which nothing else
    in the suite covers and no type checker can."""

    def case(self):
        blocks, vertices = brick((3, 3, 3))
        _, mesh_geometry, scene = make(blocks, vertices)
        return problem(scene, sources.build(mesh_geometry.cells.volume), uniform_boundary(Kind.ABSORB))

    def test_the_ceiling_is_a_width_that_still_fits_the_pool(self):
        case = self.case()

        def lower(lanes):
            tally = transport.Tally.empty(jax.random.key(0), case, lanes)
            # pyrefly: ignore[missing-attribute] - it does not model what filter_jit returns
            return transport.extend.lower(tally, case, jnp.asarray(1 << 20), 400, 8, 1e-8)

        lanes = tuning.lane_capacity(lower)
        if lanes is None:
            pytest.skip("the device does not report its memory")

        assert lanes >= tuning.MIN_LANES
        # The ceiling is extrapolated from two probe widths, so the check is
        # that the march XLA compiles at it really does claim to fit.
        limit = jax.local_devices()[0].memory_stats()["bytes_limit"]
        assert tuning.resident_bytes(lower, lanes) <= limit

    def test_the_default_width_is_the_device_and_not_the_ray_count(self):
        width = tuning.device_width()
        if width is None:
            pytest.skip("the device does not report its cores")

        # A march costs `lanes x steps` whatever is alive, so a batch that grows
        # with the ray count is what the default exists to prevent.
        assert tuning.MIN_LANES <= width < 1 << 22

        case = self.case()
        tally = transport.trace(jax.random.key(0), case, num_rays=4 * width, max_steps=400)
        deposit = tally.estimate(case)
        assert tally.rays.cell.shape == (width,)
        assert float(deposit.cells.sum() + deposit.faces.sum()) == approx(float(case.sources.total), rel=1e-6)

    def test_a_width_the_caller_passes_is_the_width_marched(self):
        """Only an automatic width is adapted. A batch the caller sized is run as
        asked, so an overrun reaches them instead of being narrowed behind them."""
        case = self.case()
        tally = transport.trace(jax.random.key(0), case, num_rays=20_000, lanes=2_000, max_steps=400)
        deposit = tally.estimate(case)
        assert tally.rays.cell.shape == (2_000,)
        assert float(deposit.cells.sum() + deposit.faces.sum()) == approx(float(case.sources.total), rel=1e-6)

    def test_a_width_past_the_ray_count_leaves_lanes_empty(self):
        """The lane count does not depend on the ray count. The lanes past the
        budget never receive a ray, and must not add energy to the balance."""
        case = self.case()
        tally = transport.trace(jax.random.key(0), case, num_rays=5_000, lanes=20_000, max_steps=400)
        deposit = tally.estimate(case)
        assert tally.rays.cell.shape == (20_000,)
        assert int(tally.launched) == 5_000
        assert float(deposit.cells.sum() + deposit.faces.sum()) == approx(float(case.sources.total), rel=1e-6)


class TestReducedState:
    """A 2D mesh is read as extruded out of plane. The component that points out
    of it is not stored -- it survives as the deficit in |tangent| -- so these
    guard the two things that makes true."""

    def launch(self, kind, kappa, num=2000, seed=0):
        blocks, vertices = brick((3, 3), dim=2)
        _, mesh_geometry, scene = make(blocks, vertices)
        case = problem(scene, sources.build(mesh_geometry.cells.volume), uniform_boundary(kind), kappa)
        return case, transport.launch(jax.random.key(seed), case, num)

    def test_specular_march_preserves_the_out_of_plane_share(self):
        """Crossings and mirror bounces are the whole march, and neither may move
        energy between the plane and the direction out of it."""
        case, rays = self.launch(Kind.SPECULAR, 0.5)
        final, _ = transport.march(jax.random.key(1), rays, case, steps=200)
        before = jnp.linalg.vector_norm(rays.state.tangent, axis=-1)
        after = jnp.linalg.vector_norm(final.state.tangent, axis=-1)
        assert np.allclose(np.asarray(before), np.asarray(after), rtol=1e-5)

    def test_travel_is_the_three_dimensional_length(self):
        """`travel` is the distance flown in 3D, which is the in-plane distance
        divided by |tangent|. Rays fly straight out, so the two are comparable."""
        case, rays = self.launch(Kind.ESCAPE, 0.0)
        final, _ = transport.march(jax.random.key(1), rays, case, steps=100)
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
        """A ray only reaches r = 0 with no angular momentum at all, which is measure zero.
        Any deposit on the axis group is a leak through the symmetry plane."""
        mesh_geometry, groups, scene = self.revolved()
        table = uniform_boundary(Kind.ABSORB)
        _, deposit = run(
            scene, sources.build(mesh_geometry.cells.volume), table, path=AxisymmetricPath(), max_steps=600
        )

        on_axis = np.flatnonzero(groups.face_group == groups.index("axis"))
        assert float(np.asarray(deposit.faces)[on_axis].sum()) == approx(0.0)
        assert float(deposit.faces.sum()) > 0, "the other walls must still absorb"
