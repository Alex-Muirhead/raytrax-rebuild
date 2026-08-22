# Examples

## Setup

The meshes are generated, not committed:

```
uv run python examples/meshes/generate.py            # both
uv run python examples/meshes/generate.py plane      # or just one
uv run python examples/meshes/generate.py --mesh-size 0.15
```

## `tracing.py`

A marimo notebook: a cold absorbing gas heated by a hot wall at the low-x end.

```
uv run marimo edit examples/tracing.py
```

Pick the domain, absorptivity and ray count from the controls at the top.

| | |
|---|---|
| **cylinder** | 3D tetrahedra, ~35k cells. Mirroring the lateral surface makes it an infinite slab, so the axial profile can be checked against `2[E₃(κa) − E₃(κb)]/(b−a)` and every radial shell should be flat. |
| **plane** | 2D, **mixed**: quads on the lower half, triangles on the upper, sharing an interior face. Shows the heating field per cell and the ray chords. |

The plane is genuinely two-dimensional radiation — directions are drawn from the
circle, not projected from a sphere — so it is *not* a slice through a 3D
problem and will not reproduce the 3D slab result. Planar and axisymmetric
treatments are future work.

A mirrored box keeps a few rays bouncing long after they carry any energy, so
judge convergence from the energy accounting rather than the count of rays that
ran out of steps: on the plane, raising `max_steps` fifteen-fold cuts that count
from 3,446 to 232 and moves the answer by nothing (5.4e-6 either way).

The last cell writes a `.vtu` next to the mesh for ParaView.
