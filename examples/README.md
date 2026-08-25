# Examples

Three marimo notebooks. Two share one problem -- a cold absorbing gas heated by
a hot wall, on a 2D plane or a 3D cylinder -- and the third is a re-entry shock
layer. Each notebook explains its own setup; this is only the map.

Meshes are generated, not committed, so run this first:

```
uv run python examples/meshes/generate.py
```

`tracing.py` is the tutorial: one problem read three ways -- 3D tets, a mixed
2D plane, and that same plane revolved -- each checked against the tangent-slab
solution. Matplotlib only.

```
uv run marimo edit examples/tracing.py
```

`visualise.py` turns the same problem into slide material: pyvista renders
written to PNG, and the axial and radial profiles written to CSV alongside the
analytic value. It needs the optional pyvista stack and a display.

```
uv sync --extra visualisation
uv run marimo edit examples/visualise.py    # "Write every figure and CSV" fills examples/figures
```

`reentry.py` is the shock layer around a Stardust-like capsule, solved
axisymmetrically, with the shock boundary radiating as a black surface at
10,000 K. Matplotlib only, so unlike `visualise.py` it wants no pyvista and no
display.

```
uv run marimo edit examples/reentry.py
```
