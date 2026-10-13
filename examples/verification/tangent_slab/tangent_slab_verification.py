import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path

    import jax
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import PolyCollection

    from raytrax import transport
    from raytrax_examples import colours, export, profiles, render
    from raytrax_examples.analytic import tangent_slab_point
    from raytrax_examples.cases import cylinder, slab
    from raytrax_examples.typst import figure

    HERE = Path(__file__).parent
    DATA = HERE / "data"
    FIGURES = HERE / "figures.typ"
    DATA.mkdir(exist_ok=True)

    CELL_EDGE = "#26262699"


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
      shows individual cells.

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


@app.cell
def _(kappa):
    slab_planar = slab.case(kappa=kappa.value)
    slab_axisymmetric = slab.case(axisymmetric=True, kappa=kappa.value)
    cylinder_fine = cylinder.case(kappa=kappa.value)
    cylinder_coarse = cylinder.case(kappa=kappa.value, mesh_size=0.6)
    return cylinder_coarse, cylinder_fine, slab_axisymmetric, slab_planar


@app.function
def solve(case, rays):
    """Trace one case, and give the volumetric heating of each cell."""
    tally = transport.trace(jax.random.key(4), case.problem, num_rays=rays, max_steps=4000)
    return np.asarray(tally.estimate(case.problem).cells) / case.geom.cells.volume


@app.cell
def _(
    cylinder_coarse,
    cylinder_fine,
    num_rays,
    slab_axisymmetric,
    slab_planar,
):
    rays = int(num_rays.value)
    heating_planar = solve(slab_planar, rays)
    heating_axisymmetric = solve(slab_axisymmetric, rays)
    heating_cylinder = solve(cylinder_fine, rays)
    heating_coarse = solve(cylinder_coarse, rays)
    return (
        heating_axisymmetric,
        heating_coarse,
        heating_cylinder,
        heating_planar,
    )


@app.cell
def _(
    cylinder_coarse,
    cylinder_fine,
    heating_axisymmetric,
    heating_coarse,
    heating_cylinder,
    heating_planar,
    kappa,
    slab_axisymmetric,
    slab_planar,
):
    errors = [
        {
            "case": name,
            "cells": int(case.mesh.topology.num_cells),
            "cell_rmse": profiles.cell_error(case, heating, kappa.value),
            "binned_rmse": profiles.profile_error(case, heating, kappa.value),
        }
        for name, case, heating in (
            ("slab_planar", slab_planar, heating_planar),
            ("slab_axisymmetric", slab_axisymmetric, heating_axisymmetric),
            ("cylinder", cylinder_fine, heating_cylinder),
            ("cylinder_coarse", cylinder_coarse, heating_coarse),
        )
    ]
    export.save_json(DATA / "errors.json", {"kappa": kappa.value, "cases": errors})
    mo.ui.table(errors, selection=None)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## The meshes

    A 2D panel leaves this notebook as a bare image of the cells, cropped to
    the mesh, with a JSON sidecar that tells Typst its bounds and what the
    colours mean. Typst draws the axes, the legend and the colour bar around
    it. A 3D panel is a pyvista render of the cylinder cut in half, shown as it
    is.
    """)
    return


@app.cell
def _(slab_planar):
    # Both slab readings share these cells, so one image serves both.
    slab_mesh = slab_planar.mesh
    slab_bounds = (*slab_mesh.geometry.vertices.min(axis=0).tolist(), *slab_mesh.geometry.vertices.max(axis=0).tolist())

    # One flat colour per element type, from lilaq's default cycle.
    slab_legend = [
        {"label": element.name, "count": len(rows), "colour": colour, "alpha": 0.55}
        for (element, rows, _), colour in zip(export.cell_polygons(slab_mesh), colours.CYCLE, strict=False)
    ]
    _fig, _ax = export.bare_axes(slab_bounds)
    for (_, _, _corners), _entry in zip(export.cell_polygons(slab_mesh), slab_legend, strict=False):
        _ax.add_collection(
            PolyCollection(
                list(_corners), facecolors=_entry["colour"], edgecolors=CELL_EDGE, linewidths=0.25, alpha=0.55
            )
        )
    export.save_image(DATA / "slab_mesh", _fig, slab_bounds, legend=slab_legend)

    mo.hstack([mo.image(figure(FIGURES, "mesh-slab-planar")), mo.image(figure(FIGURES, "mesh-slab-axisymmetric"))])
    return slab_bounds, slab_mesh


@app.cell
def _(cylinder_coarse, cylinder_fine, ragged):
    # The tilted camera fits the footprint of the cylinder on the view plane.
    _bounds = render.projected_extent(cylinder_coarse.mesh, render.CYLINDER_CAMERA)
    render.render(
        lambda plotter: render.draw(plotter, cylinder_coarse.mesh, ragged=ragged.value, tight_bounds=_bounds),
        DATA / "cylinder_coarse_mesh.png",
        size=render.fit_window(_bounds),
    )
    render.render(
        lambda plotter: render.draw(plotter, cylinder_fine.mesh, ragged=ragged.value, tight_bounds=_bounds),
        DATA / "cylinder_mesh.png",
        size=render.fit_window(_bounds),
    )

    mo.hstack([mo.image(figure(FIGURES, "mesh-cylinder-coarse")), mo.image(figure(FIGURES, "mesh-cylinder"))])
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Heating

    Each reading gets its own colour range.
    """)
    return


@app.cell
def _(heating_axisymmetric, heating_planar, slab_bounds, slab_mesh):
    for _name, _heating in (
        ("slab_planar_heating", heating_planar),
        ("slab_axisymmetric_heating", heating_axisymmetric),
    ):
        _norm = plt.Normalize(_heating.min(), _heating.max())
        _fig, _ax = export.bare_axes(slab_bounds)
        for _, _rows, _corners in export.cell_polygons(slab_mesh):
            # The edge colour follows the face: at zero width the gaps between
            # cells still show through as white hairlines.
            _cells = PolyCollection(
                list(_corners), cmap=colours.SEQUENTIAL, norm=_norm, linewidths=0.2, edgecolors="face"
            )
            _cells.set_array(_heating[_rows])
            _ax.add_collection(_cells)
        _bar = export.colourbar(_heating.min(), _heating.max(), colours.SEQUENTIAL)
        export.save_image(DATA / _name, _fig, slab_bounds, colourbar=_bar)

    mo.vstack(
        [mo.image(figure(FIGURES, "heating-slab-planar")), mo.image(figure(FIGURES, "heating-slab-axisymmetric"))]
    )
    return


@app.cell
def _(cylinder_fine, heating_cylinder, ragged):
    _bounds = render.projected_extent(cylinder_fine.mesh, render.CYLINDER_CAMERA)
    render.render(
        lambda plotter: render.draw(
            plotter, cylinder_fine.mesh, heating_cylinder, ragged.value, bar=False, tight_bounds=_bounds
        ),
        DATA / "cylinder_heating.png",
        size=render.fit_window(_bounds),
    )
    _bar = export.colourbar(heating_cylinder.min(), heating_cylinder.max(), colours.SEQUENTIAL)
    export.save_json(DATA / "cylinder_heating.json", {"colourbar": _bar})

    mo.image(figure(FIGURES, "heating-cylinder"))
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Profiles

    Along x, the heating is averaged over 25 bins of equal width and set
    against the tangent-slab solution. Across the axis, it is
    averaged over shells of equal radial width in four axial bands, each of
    which should be flat at the band's tangent-slab mean. A shell that holds no
    cell centroid is written as NaN and left as a gap.
    """)
    return


@app.cell
def _(
    cylinder_fine,
    heating_axisymmetric,
    heating_cylinder,
    heating_planar,
    kappa,
    slab_axisymmetric,
    slab_planar,
):
    for _name, _case, _heating in (
        ("slab_planar", slab_planar, heating_planar),
        ("slab_axisymmetric", slab_axisymmetric, heating_axisymmetric),
        ("cylinder", cylinder_fine, heating_cylinder),
    ):
        _edges, _profile, _ = profiles.axial_profile(_case, _heating, kappa.value)
        export.save_csv(DATA / f"{_name}_axial.csv", (_edges[:-1], _edges[1:], _profile), "lo,hi,heating")
        _x = np.linspace(_edges[0], _edges[-1], 601)
        export.save_csv(
            DATA / f"{_name}_tangent_slab.csv",
            (_x, tangent_slab_point(kappa.value, _edges[0], _x)),
            "x,tangent_slab",
        )

    mo.vstack(
        [
            mo.image(figure(FIGURES, "axial-slab-planar")),
            mo.image(figure(FIGURES, "axial-slab-axisymmetric")),
            mo.image(figure(FIGURES, "axial-cylinder")),
        ]
    )
    return


@app.cell
def _(
    cylinder_fine,
    heating_axisymmetric,
    heating_cylinder,
    kappa,
    slab_axisymmetric,
):
    for _name, _case, _heating in (
        ("slab_axisymmetric", slab_axisymmetric, heating_axisymmetric),
        ("cylinder", cylinder_fine, heating_cylinder),
    ):
        _shells, _bands = profiles.radial_profile(_case, _heating, kappa.value)
        # One column of shell means per axial band, and the band edges and
        # tangent-slab mean alongside.
        export.save_csv(
            DATA / f"{_name}_radial.csv",
            (_shells[:-1], _shells[1:], *(band.heating for band in _bands)),
            "lo,hi,band_1,band_2,band_3,band_4",
        )
        export.save_json(
            DATA / f"{_name}_radial.json",
            {"bands": [{"lo": band.lo, "hi": band.hi, "tangent_slab": band.exact} for band in _bands]},
        )

    mo.vstack([mo.image(figure(FIGURES, "radial-slab-axisymmetric")), mo.image(figure(FIGURES, "radial-cylinder"))])
    return


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

    The relative difference is normalised against the peak planar heating
    rather than the local value: heating decays by orders of magnitude away
    from the wall, so dividing cell by cell would blow up wherever the planar
    reading is near zero, without saying anything new about the size of the
    disagreement.
    """)
    return


@app.cell
def _(heating_axisymmetric, heating_planar):
    difference = heating_axisymmetric - heating_planar
    relative = difference / np.max(np.abs(heating_planar))

    mo.md(f"""
    | | |
    |---|---|
    | max absolute difference | {np.max(np.abs(difference)):.4f} 1/m |
    | mean absolute difference | {np.mean(np.abs(difference)):.4f} 1/m |
    | max relative to peak planar heating | {np.max(np.abs(relative)):.2%} |
    | peak planar heating | {heating_planar.max():.4f} 1/m |
    """)
    return difference, relative


@app.cell
def _(difference, relative, slab_bounds, slab_mesh):
    for _name, _values in (("difference", difference), ("relative", relative)):
        # Centred on zero, so that the neutral colour means agreement.
        _bound = float(np.max(np.abs(_values)))
        _norm = plt.Normalize(-_bound, _bound)
        _fig, _ax = export.bare_axes(slab_bounds)
        for _, _rows, _corners in export.cell_polygons(slab_mesh):
            _cells = PolyCollection(
                list(_corners), cmap=colours.DIVERGING, norm=_norm, linewidths=0.2, edgecolors="face"
            )
            _cells.set_array(_values[_rows])
            _ax.add_collection(_cells)
        _bar = export.colourbar(-_bound, _bound, colours.DIVERGING)
        export.save_image(DATA / _name, _fig, slab_bounds, colourbar=_bar)

    mo.vstack([mo.image(figure(FIGURES, "difference")), mo.image(figure(FIGURES, "relative"))])
    return


if __name__ == "__main__":
    app.run()
