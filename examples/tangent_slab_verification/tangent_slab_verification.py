import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path

    import jax
    import marimo as mo
    import numpy as np

    from raytrax import transport
    from raytrax_examples import cases, colours, export, profiles, render
    from raytrax_examples.typst import compile_figures

    HERE = Path(__file__).parent
    DATA = HERE / "data"

    # The coarse cylinder is only drawn as a mesh; the others are the verification cases.
    CASES = ("slab_planar", "slab_axisymmetric", "cylinder_coarse", "cylinder")


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Tangent-slab verification

    A hot black wall at low x radiates into a cold, grey, non-scattering gas,
    and a black wall at high x absorbs what gets through. With the lateral
    boundary mirrored the domain is an infinite slab, so the heating has a
    closed form in `x` alone, and every reading of every mesh must reproduce
    it.

    - **slab_planar** and **slab_axisymmetric** - one jumbled 2D mesh,
      triangles scattered among quads at varying cell size, read two different
      ways: extruded flat, and revolved about `y = 0` as the `(x, r)`
      half-plane.
    - **cylinder** - 3D tetrahedra, whose curved lateral surface is the mirror.
    - **cylinder_coarse** - the same, coarse enough that a cut through it
      shows individual cells. It is drawn, not measured.

    Two errors against the closed form are written for each case, both
    divided by the mean heating the domain would have had if every joule
    landed uniformly: the volume-weighted RMS over every cell, and the RMS
    over the 25 axial bins that the profile figure draws.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Why these meshes

    `slab` is the mixed-element case: triangles are scattered among quads
    rather than split off into their own half, the cell size varies across the
    domain, and it is conformal throughout. The two `slab_*` cases are the
    same 1,593 cells either way, which is what makes the planar-against-
    axisymmetric comparison below a direct subtraction.

    The full `cylinder` is 34,507 tets, and a ragged half-cut of that many is a
    grey smear at figure size; at 2,711 the individual cells of
    `cylinder_coarse` survive on the cut face.

    ## Running this notebook

    The 3D renders need the optional pyvista stack, and a display: a headless
    machine needs `xvfb-run`, or the `vtk-osmesa` wheel in place of `vtk`.

    ```
    uv sync --group visualisation
    ```
    """)
    return


@app.cell
def _():
    kappa = mo.ui.slider(0.1, 5.0, 0.1, value=1.0, label="Absorptivity kappa")
    num_rays = mo.ui.dropdown(["50000", "200000", "1000000"], value="1000000", label="Rays")
    ragged = mo.ui.checkbox(True, label="Ragged cut (whole cells)")
    mo.hstack([kappa, num_rays, ragged], justify="start", gap=2)
    return kappa, num_rays, ragged


@app.function
def solve(name, absorptivity, rays):
    """Trace one case, returning it with the volumetric heating of each cell."""
    case = cases.prepare(name, absorptivity)
    tally = transport.trace(jax.random.key(4), case.problem, num_rays=rays, max_steps=4000)
    return case, np.asarray(tally.estimate(case.problem).cells) / case.geom.cells.volume


@app.cell
def _(kappa, num_rays):
    results = {name: solve(name, kappa.value, int(num_rays.value)) for name in CASES}
    return (results,)


@app.cell(hide_code=True)
def _(kappa, results):
    def _errors():
        header = "| case | cells | per-cell RMSE | binned RMSE |\n|---|---|---|---|"
        rows = [
            f"| {name} | {case.mesh.topology.num_cells:,} "
            f"| {profiles.cell_error(case, heating, kappa.value):.2e} "
            f"| {profiles.profile_error(case, heating, kappa.value):.2e} |"
            for name, (case, heating) in results.items()
        ]
        return mo.md("\n".join([header, *rows]))

    _errors()
    return


@app.cell
def _(results):
    def compare(planar, axisymmetric):
        """Signed difference, and that difference as a fraction of the peak planar value.

        Normalised against the peak rather than the local value: heating decays
        by orders of magnitude away from the wall, so dividing cell by cell
        would blow up wherever the planar reading is near zero, without saying
        anything new about the size of the disagreement.
        """
        diff = axisymmetric - planar
        peak = float(np.max(np.abs(planar)))
        return diff, diff / peak

    planar_case, planar_heating = results["slab_planar"]
    _, axisym_heating = results["slab_axisymmetric"]
    diff, relative = compare(planar_heating, axisym_heating)
    return axisym_heating, diff, planar_case, planar_heating, relative


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Planar against axisymmetric

    `slab_planar` and `slab_axisymmetric` are the same cells, the same trace,
    the same everything except how the geometry is read. Cell for cell, in the
    same order, so the two heating fields subtract directly with no
    interpolation.

    Any difference is the axisymmetric geometry, not sampling noise: revolving
    the mesh turns each `y` into a radius, so a cell's volume is its flat area
    times the ring it sweeps, vanishing towards `y = 0` rather than staying
    uniform the way the planar reading treats it. Heating divides deposited
    energy by that volume, so cells nearest the axis are where the two
    readings are most exposed to disagreeing.
    """)
    return


@app.cell(hide_code=True)
def _(diff, planar_heating, relative):
    mo.md(f"""
    | | |
    |---|---|
    | max absolute difference | {np.max(np.abs(diff)):.4f} 1/m |
    | mean absolute difference | {np.mean(np.abs(diff)):.4f} 1/m |
    | max relative to peak planar heating | {np.max(np.abs(relative)):.2%} |
    | peak planar heating | {planar_heating.max():.4f} 1/m |
    """)
    return


@app.cell(hide_code=True)
def _(
    axisym_heating,
    diff,
    kappa,
    planar_case,
    planar_heating,
    ragged,
    relative,
    results,
):
    def _colourbar(values, cmap=colours.SEQUENTIAL, label="Volumetric heating", unit="/m", symmetric=False):
        # The hot end emits sigma T^4 = 1 per unit area, so deposits carry units
        # of face measure; over cell volume, heating comes out in 1/m.
        if symmetric:
            bound = float(np.max(np.abs(values)))
            return export.colourbar(-bound, bound, cmap, label, unit)
        return export.colourbar(float(values.min()), float(values.max()), cmap, label, unit)

    def _export_case(name, case, heating):
        """The mesh and heating renders, the profiles, and the errors of one case."""
        out = DATA / name
        mesh = case.mesh
        three_d = mesh.topology.dim == 3
        # Face on, the camera fits the world bounds exactly, so they are worth
        # writing out; tilted, it fits the footprint on the view plane instead.
        bounds = render.projected_extent(mesh, render.CYLINDER_CAMERA) if three_d else render.mesh_extent(mesh)
        size = render.fit_window(bounds)
        view = {} if three_d else export.viewbox(bounds)
        written = []

        legend = None

        def _draw_mesh(plotter):
            nonlocal legend
            legend = render.draw_mesh(plotter, mesh, ragged.value, legend=False, tight_bounds=bounds)

        out.mkdir(parents=True, exist_ok=True)
        render.render(_draw_mesh, out / "mesh.png", size=size)
        written += [out / "mesh.png", export.save_json(out / "mesh.json", {**view, "legend": legend or []})]

        render.render(
            lambda plotter: render.draw(plotter, mesh, heating, ragged.value, bar=False, tight_bounds=bounds),
            out / "heating.png",
            size=size,
        )
        written += [
            out / "heating.png",
            export.save_json(out / "heating.json", {**view, "colourbar": _colourbar(heating)}),
        ]

        edges, profile, exact = profiles.axial_profile(case, heating, kappa.value)
        written.append(
            export.save_csv(out / "axial.csv", (edges[:-1], edges[1:], profile, exact), "lo,hi,heating,tangent_slab")
        )
        radial = profiles.radial_profile(case, heating, kappa.value)
        if radial is not None:
            shells, bands = radial
            columns = np.vstack(
                [
                    np.column_stack(
                        [
                            np.full(shells.size - 1, band.lo),
                            np.full(shells.size - 1, band.hi),
                            shells[:-1],
                            shells[1:],
                            band.heating,
                            np.full(shells.size - 1, band.exact),
                        ]
                    )
                    for band in bands
                ]
            )
            written.append(
                export.save_csv(out / "radial.csv", list(columns.T), "band_lo,band_hi,lo,hi,heating,tangent_slab")
            )

        summary = {
            "cells": int(mesh.topology.num_cells),
            "kappa": kappa.value,
            "cell_rmse": profiles.cell_error(case, heating, kappa.value),
            "binned_rmse": profiles.profile_error(case, heating, kappa.value),
        }
        written.append(export.save_json(out / "summary.json", summary))
        return written

    def _export_comparison():
        """Planar against axisymmetric: same cells, same order, so a direct subtraction."""
        out = DATA / "comparison"
        out.mkdir(parents=True, exist_ok=True)
        mesh = planar_case.mesh
        bounds = render.mesh_extent(mesh)
        size = render.fit_window(bounds)
        written = []
        for stem, values, label, unit in (
            ("difference", diff, "Difference", "/m"),
            ("relative", relative, "Relative to peak planar heating", None),
        ):
            render.render(
                lambda plotter, values=values: render.draw(
                    plotter, mesh, values, cmap=colours.DIVERGING, bar=False, tight_bounds=bounds, symmetric=True
                ),
                out / f"{stem}.png",
                size=size,
            )
            meta = {
                **export.viewbox(bounds),
                "colourbar": _colourbar(values, colours.DIVERGING, label, unit, symmetric=True),
            }
            written += [out / f"{stem}.png", export.save_json(out / f"{stem}.json", meta)]

        centroid = mesh.geometry.cells.centroid
        written.append(
            export.save_csv(
                out / "comparison.csv",
                (centroid[:, 0], centroid[:, 1], planar_heating, axisym_heating, diff, relative),
                "x,y,planar,axisymmetric,diff,relative",
            )
        )
        return written

    def _export():
        written = [path for name, (case, heating) in results.items() for path in _export_case(name, case, heating)]
        written += _export_comparison()
        return written + compile_figures(HERE / "figures.typ")

    written = _export()
    return (written,)


@app.cell(hide_code=True)
def _(written):
    mo.vstack([mo.image(path) for path in written if path.parent.name == "figures"])
    return


@app.cell
def _(written):
    [(n,w) for (n,w) in enumerate(written) if w.parent.name == "figures"]
    return


@app.cell
def _(written):
    mo.image(written[32])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ```typst
    #import "@preview/lilaq:0.7.1" as lq


    ```
    """)
    return


if __name__ == "__main__":
    app.run()
