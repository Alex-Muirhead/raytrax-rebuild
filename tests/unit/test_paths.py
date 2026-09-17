"""The axisymmetric crossing, against a high-precision reference.

A straight 3D ray projects to a hyperbola in the (x, r) half-plane, so the
reference here is the projection itself, evaluated in longdouble: `r(t)` is the
square root of a quadratic and needs no hyperbola at all. The first root of
`max_i g_i(t)` is where the ray leaves a convex cell, independently of how
`AxisymmetricPath` chooses to parametrise the arc.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pytest import approx
from scipy.optimize import root_scalar

from raytrax.paths import AxisymmetricPath, LinearPath

LD = np.longdouble
PATH = AxisymmetricPath()
SEEDS = range(5)


def trace(t, ray, dtype=LD):
    """The projected trace `(x, r)` of a 3D ray, at parameter `t`."""
    x0, r0, axial, radial = (dtype(v) for v in ray)
    t = np.asarray(t, dtype=dtype)
    reduced = dtype(1) - axial**2
    return x0 + axial * t, np.sqrt(reduced * t * t + 2 * r0 * radial * t + r0**2)


def gap(t, normal, offset, ray):
    """`max_i (n_i . p(t) - c_i)`: negative inside the cell, zero on its boundary."""
    x, r = trace(t, ray)
    n, c = np.asarray(normal, LD), np.asarray(offset, LD)
    return (n[:, 0] * np.asarray(x)[..., None] + n[:, 1] * np.asarray(r)[..., None] - c).max(axis=-1)


def scanned_exit(normal, offset, ray, horizon=12.0):
    """Where the ray leaves the cell, as a 3D distance."""
    # The scan brackets the first crossing; a bracket over the whole horizon
    # could hold a later one instead.
    grid = np.linspace(LD(0), LD(horizon), 3000)
    crossed = np.flatnonzero(gap(grid, normal, offset, ray) > 0)
    assert crossed.size, "the ray never left the cell"
    bracket = (grid[crossed[0] - 1], grid[crossed[0]])
    return root_scalar(lambda t: float(gap(LD(t), normal, offset, ray)), bracket=bracket, method="brentq").root


def bounded_cell(rng, inside, num_faces=4):
    """A convex cell containing `inside`, with normals that positively span.

    Random half-spaces are frequently unbounded, and a ray then legitimately
    never exits, so the angular gap is capped to force a closed cell.
    """
    while True:
        angles = np.sort(rng.uniform(0, 2 * np.pi, num_faces))
        if np.diff(np.concatenate([angles, angles[:1] + 2 * np.pi])).max() < np.pi - 0.15:
            break
    normal = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    offset = normal @ inside + rng.uniform(0.3, 1.5, num_faces)
    return jnp.asarray(normal), jnp.asarray(offset)


def direction_meeting_axis_within(rng, ratio):
    """A 3D direction whose closest approach to the axis is `ratio` times the radius.

    The approach is `|sin(psi)|` for `psi` the transverse impact angle, which is
    what makes the angular momentum small.
    """
    axial = rng.uniform(-0.9, 0.9)
    transverse = np.sqrt(1 - axial**2)
    psi = np.arcsin(np.clip(ratio, 0.0, 1.0)) * rng.choice([-1, 1])
    return np.array([axial, rng.choice([-1, 1]) * transverse * np.cos(psi), transverse * np.sin(psi)])


def unit(rng, dim):
    """An isotropic unit vector in `dim` dimensions."""
    vector = rng.normal(size=dim)
    return vector / np.linalg.norm(vector)


@jax.jit
def cross(position, direction, normal, offset):
    state = PATH.launch(position, direction, jnp.zeros(()))
    slot, step = PATH.crossing(normal, offset, state)
    moved, _ = PATH.advance(state, step)
    return moved, slot


@jax.jit
def cross_and_bounce(position, direction, normal, offset):
    """The launched state, that state moved to its exit, and the exit mirrored off the face it left by."""
    state = PATH.launch(position, direction, jnp.zeros(()))
    slot, step = PATH.crossing(normal, offset, state)
    moved, _ = PATH.advance(state, step)
    return state, moved, PATH.reflect(moved, normal[slot])


@jax.jit
def hold(position, direction):
    """A launched ray advanced by nothing."""
    held, _ = PATH.advance(PATH.launch(position, direction, jnp.zeros(())), jnp.zeros(()))
    return held


@eqx.filter_jit
def mirror(path, position, direction, normal):
    """A launched state, and that state mirrored off a plane with `normal`."""
    state = path.launch(position, direction, jnp.zeros(()))
    return state, path.reflect(state, normal)


def sample_case(seed, ratio=None):
    """A ray inside a bounded cell, with the scanned exit it must reproduce."""
    rng = np.random.default_rng(seed)
    r0, x0 = rng.uniform(0.2, 2.0), rng.uniform(-1.0, 1.0)
    direction = unit(rng, 3) if ratio is None else direction_meeting_axis_within(rng, ratio)
    ray = (x0, r0, direction[0], direction[1])
    normal, offset = bounded_cell(rng, np.array([x0, r0]))
    return ray, direction, normal, offset, scanned_exit(normal, offset, ray)


class TestCrossing:
    """The crossing must land where the projected ray actually leaves the cell."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_matches_a_scanned_reference(self, seed):
        ray, direction, normal, offset, want = sample_case(seed)
        moved, _ = cross(jnp.array(ray[:2]), jnp.asarray(direction), normal, offset)
        expected = np.array([float(c) for c in trace(LD(want), ray)])
        assert np.linalg.norm(np.asarray(moved.position) - expected) < 1e-10

    @pytest.mark.parametrize("ratio", [1e-2, 1e-4, 1e-8, 1e-12, 1e-16])
    def test_holds_up_for_rays_that_nearly_meet_the_axis(self, ratio):
        """The angular momentum is carried, not recovered from `1 - |tangent|^2`.

        Recovering it cancels once the closest approach falls below sqrt(eps) of
        the radius, which is where a centre-anchored frame loses the crossing
        outright. Carrying it keeps full precision to any approach.
        """
        ray, direction, normal, offset, want = sample_case(0, ratio=ratio)
        moved, _ = cross(jnp.array(ray[:2]), jnp.asarray(direction), normal, offset)
        expected = np.array([float(c) for c in trace(LD(want), ray)])
        assert np.linalg.norm(np.asarray(moved.position) - expected) < 1e-9

    def test_leaves_through_the_face_it_entered_by_when_the_arc_turns(self):
        """A ray that has just entered through the face `r = 1` sits a rounding
        error outside it, heading inward. Its arc turns and leaves through that
        same face, so the face has two positive roots: the one just behind the
        ray, and the exit. Taking the smaller returns a step of about 1e-13.

        With `r0` and the direction below, the exit is the larger root of
        `(1 - axial^2) t^2 + 2 r0 radial t + r0^2 - 1 = 0`.
        """
        r0, axial, radial = 1 + 1e-13, 0.3, -0.6
        direction = jnp.array([axial, radial, np.sqrt(1 - axial**2 - radial**2)])
        normal = jnp.array([[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]])
        offset = jnp.array([1.0, 5.0, 5.0])

        reduced, linear, constant = 1 - axial**2, 2 * r0 * radial, r0**2 - 1
        exit = (-linear + np.sqrt(linear**2 - 4 * reduced * constant)) / (2 * reduced)

        moved, slot = cross(jnp.array([0.0, r0]), direction, normal, offset)
        assert int(slot) == 0
        assert float(moved.travel) == approx(exit, rel=1e-12)


class TestState:
    """What `launch` builds, and what the march must not disturb."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_the_parametrisation_starts_at_the_ray(self, seed):
        """A zero step must leave the ray exactly where it was."""
        rng = np.random.default_rng(seed)
        direction = unit(rng, 3)
        position = jnp.array([rng.uniform(-1, 1), rng.uniform(0.05, 2.0)])
        held = hold(position, jnp.asarray(direction))
        assert np.allclose(np.asarray(held.position), np.asarray(position), atol=1e-12)
        assert np.allclose(np.asarray(held.tangent), direction[:2], atol=1e-12)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_angular_momentum_survives_a_crossing_and_a_bounce(self, seed):
        """The angular momentum is the conserved quantity the whole formulation rests on."""
        ray, direction, normal, offset, _ = sample_case(seed)
        state, moved, bounced = cross_and_bounce(jnp.array(ray[:2]), jnp.asarray(direction), normal, offset)
        assert float(moved.angular_momentum) == approx(float(state.angular_momentum), rel=1e-12)
        assert float(bounced.angular_momentum) == approx(float(state.angular_momentum), rel=1e-12)
        # A bounce leaves the radius alone, so |tangent| cannot move either.
        before = float(jnp.linalg.vector_norm(moved.tangent))
        assert float(jnp.linalg.vector_norm(bounced.tangent)) == approx(before, rel=1e-12)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_travel_is_the_three_dimensional_arclength(self, seed):
        """`travel` must be the 3D distance, not the hyperbola's own parameter."""
        ray, direction, normal, offset, want = sample_case(seed)
        moved, _ = cross(jnp.array(ray[:2]), jnp.asarray(direction), normal, offset)
        assert float(moved.travel) == approx(want, rel=1e-9)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_the_axial_tangent_is_untouched_by_the_march(self, seed):
        """Only a wall can change the axial direction cosine."""
        ray, direction, normal, offset, _ = sample_case(seed)
        moved, _ = cross(jnp.array(ray[:2]), jnp.asarray(direction), normal, offset)
        assert float(moved.tangent[0]) == approx(float(direction[0]), rel=1e-12)


class TestReflect:
    """`reflect` is shared, so it must hold for the straight mode too."""

    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("path", [LinearPath(), AxisymmetricPath()])
    def test_mirroring_preserves_the_tangent_magnitude(self, path, seed):
        rng = np.random.default_rng(seed)
        direction, normal = unit(rng, 3), unit(rng, 2)
        state, bounced = mirror(path, jnp.array([0.3, 1.1]), jnp.asarray(direction), jnp.asarray(normal))
        assert float(jnp.linalg.vector_norm(bounced.tangent)) == approx(
            float(jnp.linalg.vector_norm(state.tangent)), rel=1e-12
        )
