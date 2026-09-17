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

## Geometric Verification -- Tangent Slab

The tangent slab problem has an analytical 1D solution, and each domain has
sufficient symmetry to collapse to 1D. Both "ends" of the domain have fixed
radiative intensities; one hot and one cold. The medium is nonscattering, and
has uniform absorptivity.

This can be run for 2D planar, 2D axisymmetric, and 3D with a cylindrical
domain. Symmetry boundaries are applied to the "sides" of each domain.
The heat-flux divergence is compared against the analytical solution.
In order to verify the symmetry boundary, each domain is run at two different
mesh resolution.

## Geometric Verification -- Concentric Spheres / Cylinders

The concentric spheres and cylinders problems have analytical 1D solutions.
Both the "inner" and "outer" boundaries have fixed radiative intensities; one
hot and one cold. The medium is nonscattering, and has uniform absorptivity.

The cylindrical case can be run for the 2D planar solver. The spherical case
can be run for both 2D axisymmetric and 3D solvers. The heat-flux divergence 
is compared against the analytical solution in each case.

## Sampling Convergence

Monte-Carlo simulations should improve asymptotically with increased samples.
The tangent slab case (see above) is run on each domain for increasing numbers
of samples, and the error in heat-flux divergence is plotted on a log-log plot.

The convergence order is estimated and reported for each domain (2D planar,
2D axisymmetric, and 3D). The remaining "bias" in error (after convergence)
is reported for each domain as well. 

## Spatial Order and Convergence

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

> [!NOTE]
> Meshes are generated, not committed, so run this first:
> 
> ```
> uv run python examples/meshes/generate.py
> ```
