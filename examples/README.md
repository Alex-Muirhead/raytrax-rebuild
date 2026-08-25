# Examples

Two marimo notebooks over the same problem: a cold absorbing gas heated by a hot
wall, on a 2D plane or a 3D cylinder. They document themselves; these are the
things they cannot tell you from the inside.

`tracing.py` is the tutorial, and matplotlib only.

```
uv run python examples/meshes/generate.py    # meshes are generated, not committed
uv run marimo edit examples/tracing.py
```

`visualise.py` produces slide material: pyvista renders written to PNG, and the
axial and radial profiles written to CSV alongside the analytic tangent-slab
value. It needs the optional pyvista stack, and renders off screen against a
real GL context, so it wants a display. A headless machine needs `xvfb-run` or
the `vtk-osmesa` wheel instead.

```
uv sync --extra visualisation
uv run marimo edit examples/visualise.py    # "Write every figure and CSV" fills examples/figures
```

Its `slab` mesh is the mixed-element case worth showing: triangles scattered
among quads at varying cell size, conformal throughout, and the same file reads
either as an extruded slab or, revolved about `y = 0`, as the meridional
half-plane of the cylinder. `cylinder_coarse` exists because a ragged half-cut
of the full 34,507-tet mesh is unreadable at slide size.
