"""The axisymmetric crossing, against a high-precision reference.

A straight 3D ray projects to a hyperbola in the (x, r) half-plane, so the
reference here is the projection itself, evaluated in longdouble: `r(t)` is the
square root of a quadratic and needs no hyperbola at all. Bisecting
`max_i g_i(t)` finds where the ray leaves a convex cell, independently of how
`AxisymmetricPath` chooses to parametrise the arc.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pytest import approx

from raytrax.paths import AxisymmetricPath, LinearPath

LD = np.longdouble
PATH = AxisymmetricPath()


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
    """Where the ray leaves the cell, by dense scan then bisection."""
    grid = np.linspace(LD(0), LD(horizon), 3000)[1:]
    crossed = np.flatnonzero(gap(grid, normal, offset, ray) > 0)
    if crossed.size == 0:
        return None
    hi, lo = grid[crossed[0]], (grid[crossed[0] - 1] if crossed[0] else LD(0))
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if gap(mid, normal, offset, ray) > 0 else (mid, hi)
    return float((lo + hi) / 2)


def bounded_cell(rng, inside, num_faces=4, touch_axis=False):
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
    if touch_axis:
        normal = np.vstack([normal, [0.0, -1.0]])
        offset = np.concatenate([offset, [0.0]])
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


@jax.jit
def cross(position, direction, normal, offset):
    state = PATH.launch(position, direction, jnp.zeros(()))
    slot, step = PATH.crossing(normal, offset, state)
    moved, _ = PATH.advance(state, step)
    return moved, slot


def sample_case(rng, ratio=None, touch_axis=False):
    """A ray inside a bounded cell, with the scanned exit it must reproduce."""
    r0, x0 = rng.uniform(0.2, 2.0), rng.uniform(-1.0, 1.0)
    if ratio is None:
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
    else:
        direction = direction_meeting_axis_within(rng, ratio)
    ray = (x0, r0, direction[0], direction[1])
    normal, offset = bounded_cell(rng, np.array([x0, r0]), touch_axis=touch_axis)
    return ray, direction, normal, offset, scanned_exit(normal, offset, ray)


class TestCrossing:
    """The crossing must land where the projected ray actually leaves the cell."""

    def test_matches_a_scanned_reference(self):
        rng = np.random.default_rng(0)
        errors = []
        for _ in range(400):
            ray, direction, normal, offset, want = sample_case(rng)
            if want is None or want < 1e-9:
                continue
            moved, _ = cross(jnp.array([ray[0], ray[1]]), jnp.asarray(direction), normal, offset)
            expected = np.array([float(c) for c in trace(LD(want), ray)])
            errors.append(np.linalg.norm(np.asarray(moved.position) - expected))
            assert float(moved.travel) == approx(want, rel=1e-9)
        assert len(errors) > 300, "too few cases produced a bounded exit"
        assert max(errors) < 1e-10

    @pytest.mark.parametrize("ratio", [1e-2, 1e-4, 1e-8, 1e-12, 1e-16])
    def test_holds_up_for_rays_that_nearly_meet_the_axis(self, ratio):
        """The angular momentum is carried, not recovered from `1 - |tangent|^2`.

        Recovering it cancels once the closest approach falls below sqrt(eps) of
        the radius, which is where a centre-anchored frame loses the crossing
        outright. Carrying it keeps full precision to any approach.
        """
        rng = np.random.default_rng(hash(ratio) % 2**31)
        errors = []
        for _ in range(120):
            ray, direction, normal, offset, want = sample_case(rng, ratio=ratio)
            if want is None or want < 1e-9:
                continue
            moved, _ = cross(jnp.array([ray[0], ray[1]]), jnp.asarray(direction), normal, offset)
            expected = np.array([float(c) for c in trace(LD(want), ray)])
            errors.append(np.linalg.norm(np.asarray(moved.position) - expected))
        assert len(errors) > 80
        assert max(errors) < 1e-9

    def test_leaves_through_the_face_it_entered_by_when_the_arc_turns(self):
        """A cell against the axis can be entered and left through one face.

        The arc turns at its closest approach and bends back out, so the second
        root on that same plane is the exit. Selecting by ordering alone returns
        the crossing back into the cell, which skips it and charges its length
        to the previous one.
        """
        rng = np.random.default_rng(7)
        turned = 0
        for _ in range(400):
            ray, direction, normal, offset, want = sample_case(rng, touch_axis=True)
            if want is None or want < 1e-9:
                continue
            # Did the arc turn before leaving? The radius bottoms out inside.
            probe = np.linspace(0.0, want, 200)
            _, radius = trace(probe, ray)
            if not (0 < int(np.argmin(radius)) < len(probe) - 1):
                continue
            turned += 1
            moved, _ = cross(jnp.array([ray[0], ray[1]]), jnp.asarray(direction), normal, offset)
            expected = np.array([float(c) for c in trace(LD(want), ray)])
            assert np.linalg.norm(np.asarray(moved.position) - expected) < 1e-10
        assert turned > 20, f"only {turned} cases actually turned inside the cell"


class TestState:
    """What `launch` builds, and what the march must not disturb."""

    def test_the_parametrisation_starts_at_the_ray(self):
        """A zero step must leave the ray exactly where it was."""
        rng = np.random.default_rng(3)
        for _ in range(200):
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            position = jnp.array([rng.uniform(-1, 1), rng.uniform(0.05, 2.0)])
            state = PATH.launch(position, jnp.asarray(direction), jnp.zeros(()))
            held, _ = PATH.advance(state, jnp.zeros(()))
            assert np.allclose(np.asarray(held.position), np.asarray(position), atol=1e-12)
            assert np.allclose(np.asarray(held.tangent), np.asarray(direction[:2]), atol=1e-12)

    def test_angular_momentum_survives_a_crossing_and_a_bounce(self):
        """The angular momentum is the conserved quantity the whole formulation rests on."""
        rng = np.random.default_rng(11)
        for _ in range(200):
            ray, direction, normal, offset, want = sample_case(rng)
            if want is None or want < 1e-9:
                continue
            state = PATH.launch(jnp.array([ray[0], ray[1]]), jnp.asarray(direction), jnp.zeros(()))
            slot, step = PATH.crossing(normal, offset, state)
            moved, _ = PATH.advance(state, step)
            bounced = PATH.reflect(moved, normal[slot])
            assert float(moved.angular_momentum) == approx(float(state.angular_momentum), rel=1e-12)
            assert float(bounced.angular_momentum) == approx(float(state.angular_momentum), rel=1e-12)
            # A bounce leaves the radius alone, so |tangent| cannot move either.
            before = float(jnp.linalg.vector_norm(moved.tangent))
            assert float(jnp.linalg.vector_norm(bounced.tangent)) == approx(before, rel=1e-12)

    def test_travel_is_the_three_dimensional_arclength(self):
        """`travel` must be the 3D distance, not the hyperbola's own parameter."""
        rng = np.random.default_rng(5)
        for _ in range(200):
            ray, direction, normal, offset, want = sample_case(rng)
            if want is None or want < 1e-9:
                continue
            moved, _ = cross(jnp.array([ray[0], ray[1]]), jnp.asarray(direction), normal, offset)
            assert float(moved.travel) == approx(want, rel=1e-9)

    def test_the_axial_tangent_is_untouched_by_the_march(self):
        """Only a wall can change the axial direction cosine."""
        rng = np.random.default_rng(13)
        for _ in range(200):
            ray, direction, normal, offset, want = sample_case(rng)
            if want is None or want < 1e-9:
                continue
            moved, _ = cross(jnp.array([ray[0], ray[1]]), jnp.asarray(direction), normal, offset)
            assert float(moved.tangent[0]) == approx(float(direction[0]), rel=1e-12)


class TestReflect:
    """`reflect` is shared, so it must hold for the straight mode too."""

    @pytest.mark.parametrize("path", [LinearPath(), AxisymmetricPath()])
    def test_mirroring_preserves_the_tangent_magnitude(self, path):
        rng = np.random.default_rng(17)
        for _ in range(100):
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            normal = rng.normal(size=2)
            normal /= np.linalg.norm(normal)
            state = path.launch(jnp.array([0.3, 1.1]), jnp.asarray(direction), jnp.zeros(()))
            bounced = path.reflect(state, jnp.asarray(normal))
            assert float(jnp.linalg.vector_norm(bounced.tangent)) == approx(
                float(jnp.linalg.vector_norm(state.tangent)), rel=1e-12
            )
