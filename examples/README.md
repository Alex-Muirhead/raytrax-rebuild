# Examples

Each example is a self-contained directory, for demonstrating isolated
features of the solver in the corresponding section of the thesis. Verification
cases are built on top of analytical solutions (e.g. tangent slab, concentric
spheres and cylinders).

The Python code for each example is contained within a single marimo notebook.
Visualisations are built in a corresponding Typst document for consistent
styling, and are exported as `png` files for viewing in the marimo notebook.
If results are shown in 3D, they are first generated in Pyvista, then annotated
in Typst, and finally exported. As such, the marimo notebooks will often
generate corresponding `.csv` and `.json` files for each result, in a `./data`
directory in the example.

## Running

Each notebook regenerates its `data/` and compiles its `figures.typ` into
`figures/`, one `png` per figure for the notebook to show, every time it runs.
Both directories are generated, not committed. The thesis draws the figures
itself from `data/`, so nothing else is exported. Run one headless with

```
uv run python examples/<name>/<name>.py
```

or open it with `uv run marimo edit examples/<name>/<name>.py`.

Each problem the examples run is one module in
`examples/src/raytrax_examples/cases/`: `slab`, `plane`, `cylinder` and
`reentry`. A module builds its own mesh with gmsh and its own `Problem`, so a
new case is a new file and leaves the others alone. Meshes are generated the
first time a case asks for one, and cached under `examples/meshes/`; delete a
`.msh` there to build it again.

Code shared between examples is the `raytrax-examples` package under
`examples/src/`, a uv workspace member that `uv sync` installs with the dev
group. It is not part of `raytrax` itself. The 3D renders need pyvista, from
`uv sync --group visualisation`, and a display.

Every `figures.typ` defines one function per figure, styled by `style.typ`. The
thesis imports one directly, compiled with a `--root` that contains this
repository:

```typst
#import "/examples/style.typ": number-setup
#import "/examples/reentry/figures.typ": wall-flux
#number-setup
#figure(wall-flux(), caption: [...])
```

Numbers and units in the figures are formatted by `zero`, and follow the
document's `set-num`, `set-group` and `set-unit` settings. `number-setup` is the
configuration the standalone figures use; the thesis can place it once, or set
its own.

## Geometric Verification -- Tangent Slab -- `tangent_slab_verification/`

The tangent slab problem has an analytical 1D solution, and each domain has
sufficient symmetry to collapse to 1D. Both "ends" of the domain have fixed
radiative intensities; one hot and one cold. The medium is nonscattering, and
has uniform absorptivity.

This can be run for 2D planar, 2D axisymmetric, and 3D with a cylindrical
domain. Symmetry boundaries are applied to the "sides" of each domain.
The heat-flux divergence is compared against the analytical solution.
In order to verify the symmetry boundary, each domain is run at two different
mesh resolution.

<!-- Not yet done: the 3D cylinder runs at two resolutions, but the 2D slab runs at -->
<!-- one. -->

## Geometric Verification -- Concentric Spheres / Cylinders -- `concentric_verification/`

<!-- Not yet implemented. -->

The concentric spheres and cylinders problems have analytical 1D solutions.
Both the "inner" and "outer" boundaries have fixed radiative intensities; one
hot and one cold. The medium is nonscattering, and has uniform absorptivity.

The cylindrical case can be run for the 2D planar solver. The spherical case
can be run for both 2D axisymmetric and 3D solvers. The heat-flux divergence 
is compared against the analytical solution in each case.

## Sampling Convergence -- `sampling_convergence/`

Monte-Carlo simulations should improve asymptotically with increased samples.
The tangent slab case (see above) is run on each domain for increasing numbers
of samples, and the error in heat-flux divergence is plotted on a log-log plot.

The convergence order is estimated and reported for each domain (2D planar,
2D axisymmetric, and 3D). The remaining "bias" in error (after convergence)
is reported for each domain as well. 

<!-- Not yet done: the error curves are written and plotted, but the order and the -->
<!-- bias are not yet estimated. A full sweep is 75 traces, about six minutes on a -->
<!-- GPU; `-- --largest 100000` after the script path caps the ray count. -->

## Spatial Order and Convergence -- `spatial_convergence/`

<!-- Not yet implemented. -->

After convergence with increased samples, the remaining error is considered
"bias". One source of this bias is error from the spatial order of the
implementation and the spatial resolution of the mesh.

For the tangent slab case, a mesh convergence study is run for
- Each 2D domain (planar and axisymmetric), and
- Each number of samples (from the above sampling convergence).
The latter is done as the number of samples needed to reach convergence is
dependent on the number / resolution of the cells.

The mesh convergence study begins with a coarse mesh, and repeatedly halves
the average diameter of the cells within the mesh, until convergence is reached
or until the RSME error in heat-flux divergence drops below a tolerance.

## Re-entry Shock Layer -- `reentry/`

The shock layer around a Stardust-like capsule, solved axisymmetrically, with
the shock radiating as a black surface at 10,000 K into cold grey gas. There is
no closed form; the energy budget has to balance, and the result is the
radiative load along the capsule wall.

## Performance -- `performance/`

Throughput and compile time of the march against the number of lanes, on the
2D planar, 2D axisymmetric, 3D and re-entry meshes. `-- --to-power 16` after the
script path shortens the ladder. For one case at one lane count, use
`uv run python -m raytrax_examples.benchmark`; the profilers in `scripts/`
break a run down further.
