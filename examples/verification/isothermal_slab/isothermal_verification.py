import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import itertools
    from pathlib import Path

    import jax
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import PolyCollection
    from scipy.constants import Stefan_Boltzmann

    from raytrax import transport
    from raytrax.interactions import Kind
    from raytrax_examples import colours, export, profiles
    from raytrax_examples.analytic import isothermal_slab, isothermal_slab_point
    from raytrax_examples.cases import square
    from raytrax_examples.typst import figure

    HERE = Path(__file__).parent
    DATA = HERE / "data"
    FIGURES = HERE / "figures.typ"
    DATA.mkdir(exist_ok=True)

    CELL_EDGE = "#26262699"

    KAPPA = 1.0
    # Incident radiation leaves this notebook in MW/m^2.
    SCALE = 1e-6
    EMISSIVE_POWER = Stefan_Boltzmann * square.TEMPERATURE**4
    READINGS = {"planar": False, "axisymmetric": True}
    BOUNDARIES = {"specular": Kind.SPECULAR, "absorbing": Kind.ABSORB}
    RESOLUTIONS = (15, 30, 45)
    # The band in y (r when revolved) whose cells are compared with the slab.
    STRIP = (5.0, 10.0)
    # The top of the shared colour range of the incident-radiation maps.
    HIGH_PERCENTILE = 99.5


@app.cell(hide_code=True)
def _():
    mo.md(f"""
    # Isothermal-slab verification

    A hot, grey, non-scattering gas at {square.TEMPERATURE:g} K, with
    kappa = {KAPPA:g} 1/m, fills a {square.SIDE:g} m square between two cold
    black walls at x = 0 and x = {square.SIDE:g} m. Every joule starts in the
    gas, so the walls see only what escapes it.

    The top and bottom of the square are either **specular**, which makes the
    square an infinite slab, or **absorbing**, which makes it a black
    enclosure on all four sides. Each runs two ways: extruded flat, and
    revolved about `y = 0` as the `(x, r)` half-plane of a cylinder, on
    structured grids of {", ".join(f"{n}x{n}" for n in RESOLUTIONS)} quads.

    The incident radiation is the deposit of a cell over `kappa V`. In an
    infinite slab it has a closed form in `x` alone, which reaches
    `4 sigma T^4` = {4 * EMISSIVE_POWER * SCALE:.3f} MW/m^2 deep in the gas.
    Both boundary kinds are compared against it over the strip
    {STRIP[0]:g} <= y <= {STRIP[1]:g}: there the top and bottom are at least
    {KAPPA * STRIP[0]:g} optical depths away, so an absorbing side changes
    nothing measurable, and a mirror that returns the wrong energy shows up
    as a departure in the specular case.
    """)
    return


@app.cell
def _():
    num_rays = mo.ui.dropdown(["200000", "1000000", "5000000"], value="1000000", label="Rays")
    mo.hstack([num_rays], justify="start")
    return (num_rays,)


@app.cell
def _():
    cases = {
        (reading, boundary, cells): square.case(axisymmetric=axisymmetric, lateral=kind, kappa=KAPPA, cells=cells)
        for (reading, axisymmetric), (boundary, kind), cells in itertools.product(
            READINGS.items(), BOUNDARIES.items(), RESOLUTIONS
        )
    }
    return (cases,)


@app.function
def panels(name):
    """Show one figure per reading and boundary kind, two by two, each under its label."""
    return mo.vstack(
        [
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md(f"**{reading.capitalize()}, {boundary}**"),
                            mo.image(figure(FIGURES, f"{name}-{reading}-{boundary}")),
                        ]
                    )
                    for boundary in BOUNDARIES
                ]
            )
            for reading in READINGS
        ]
    )


@app.function
def solve(case, rays):
    """Trace one case, and give the incident radiation of each cell in MW/m^2."""
    tally = transport.trace(jax.random.key(4), case.problem, num_rays=rays, max_steps=4000)
    return SCALE * np.asarray(tally.estimate(case.problem).cells) / (KAPPA * case.geom.cells.volume)


@app.function
def strip_profile(case, incident):
    """Give the strip mean and spread of each column of cells, and the slab mean over the same bins, in MW/m^2.

    Each row of cells in the strip is one sample of the same profile, so the
    spread is the standard deviation down a column.
    """
    centroid, volume = case.mesh.geometry.cells.centroid, case.geom.cells.volume
    inside = (centroid[:, 1] > STRIP[0]) & (centroid[:, 1] < STRIP[1])
    # One bin per column of the structured grid.
    edges = np.unique(case.mesh.geometry.vertices[:, 0].round(9))
    position, values, weight = centroid[inside, 0], incident[inside], volume[inside]
    profile = profiles.rebin(position, values, weight, edges)
    spread = np.sqrt(np.maximum(profiles.rebin(position, values**2, weight, edges) - profile**2, 0.0))
    return edges, profile, spread, SCALE * EMISSIVE_POWER * isothermal_slab(KAPPA, square.SIDE, edges)


@app.cell
def _(cases, num_rays):
    incident = {}
    with mo.status.progress_bar(total=len(cases), title="Tracing") as bar:
        for (_reading, _boundary, _cells), _case in cases.items():
            bar.update(increment=0, subtitle=f"{_reading}, {_boundary}, {_cells} x {_cells}")
            incident[_reading, _boundary, _cells] = solve(_case, int(num_rays.value))
            bar.update()
    return (incident,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Error over the strip

    The RMS and the largest difference between the strip profile and the
    slab, each divided by `4 sigma T^4`.
    """)
    return


@app.cell
def _(cases, incident):
    peak = 4 * EMISSIVE_POWER * SCALE
    errors = []
    for (_reading, _boundary, _cells), _case in cases.items():
        _, _profile, _, _exact = strip_profile(_case, incident[_reading, _boundary, _cells])
        errors.append(
            {
                "reading": _reading,
                "boundary": _boundary,
                "cells": _cells,
                "strip_rmse": float(np.sqrt(np.mean((_profile - _exact) ** 2)) / peak),
                "strip_max": float(np.max(np.abs(_profile - _exact)) / peak),
            }
        )
    export.save_json(
        DATA / "errors.json", {"kappa": KAPPA, "temperature": square.TEMPERATURE, "strip": STRIP, "cases": errors}
    )
    mo.ui.table(errors, selection=None)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## The meshes

    Each grid is drawn with the strip that the profiles average over.
    """)
    return


@app.cell
def _(cases):
    bounds = (0.0, 0.0, square.SIDE, square.SIDE)
    for _cells in RESOLUTIONS:
        _mesh = cases["planar", "specular", _cells].mesh
        _fig, _ax = export.bare_axes(bounds)
        for _, _, _corners in export.cell_polygons(_mesh):
            _ax.add_collection(
                PolyCollection(
                    list(_corners), facecolors=colours.CYCLE[0], edgecolors=CELL_EDGE, linewidths=0.4, alpha=0.55
                )
            )
        export.save_image(DATA / f"mesh_{_cells}", _fig, bounds, strip=STRIP)

    mo.hstack(
        [
            mo.vstack([mo.md(f"**{_cells} x {_cells}**"), mo.image(figure(FIGURES, f"mesh-{_cells}"))])
            for _cells in RESOLUTIONS
        ]
    )
    return (bounds,)


@app.cell(hide_code=True)
def _():
    mo.md(f"""
    ## Incident radiation

    The four combinations on the {RESOLUTIONS[-1]}x{RESOLUTIONS[-1]} grid, on
    one shared colour range. With mirrors the field depends on `x` alone;
    with absorbing sides it falls towards every edge, and the strip is where
    it does not.

    The range runs from the lowest cell to the {HIGH_PERCENTILE:g}th
    percentile. Above that are only the smallest revolved cells, next to the
    axis, which see the fewest rays: left in, their noise stretches the range
    and pushes every other cell into the middle of it. They saturate instead.
    """)
    return


@app.cell
def _(bounds, cases, incident):
    finest = RESOLUTIONS[-1]
    _maps = {
        (_reading, _boundary): incident[_reading, _boundary, finest]
        for _reading in READINGS
        for _boundary in BOUNDARIES
    }
    _every = np.concatenate(list(_maps.values()))
    _low, _high = float(_every.min()), float(np.percentile(_every, HIGH_PERCENTILE))
    _norm = plt.Normalize(_low, _high)
    for (_reading, _boundary), _values in _maps.items():
        _fig, _ax = export.bare_axes(bounds)
        for _, _rows, _corners in export.cell_polygons(cases[_reading, _boundary, finest].mesh):
            # The edge colour follows the face: at zero width the gaps between
            # cells still show through as white hairlines.
            _patches = PolyCollection(
                list(_corners), cmap=colours.SEQUENTIAL, norm=_norm, linewidths=0.2, edgecolors="face"
            )
            _patches.set_array(_values[_rows])
            _ax.add_collection(_patches)
        _bar = export.colourbar(_low, _high, colours.SEQUENTIAL)
        export.save_image(DATA / f"incident_{_reading}_{_boundary}", _fig, bounds, colourbar=_bar, strip=STRIP)

    panels("incident")
    return (finest,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Strip profiles

    Each row of cells inside the strip is one sample of the same profile in
    `x`. Each grid is a step line through the mean of each column of cells,
    with an error bar of one standard deviation down the column at the bin
    centre. The dashed line is the slab point solution.
    """)
    return


@app.cell
def _(cases, incident):
    _x = np.linspace(0.0, square.SIDE, 601)
    export.save_csv(
        DATA / "slab.csv", (_x, SCALE * EMISSIVE_POWER * isothermal_slab_point(KAPPA, square.SIDE, _x)), "x,incident"
    )
    for (_reading, _boundary, _cells), _case in cases.items():
        _edges, _profile, _spread, _exact = strip_profile(_case, incident[_reading, _boundary, _cells])
        export.save_csv(
            DATA / f"strip_{_reading}_{_boundary}_{_cells}.csv",
            (_edges[:-1], _edges[1:], _profile, _spread, _exact),
            "lo,hi,incident,spread,slab",
        )

    panels("strip")
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Planar against axisymmetric

    With mirrors, both readings are the same infinite slab, on the same cells
    in the same order, so the two fields subtract directly. What is left
    should be sampling noise, largest towards `y = 0` where the revolved
    cells are smallest and see the fewest rays.
    """)
    return


@app.cell
def _(bounds, cases, finest, incident):
    difference = incident["axisymmetric", "specular", finest] - incident["planar", "specular", finest]
    # Centred on zero, so that the neutral colour means agreement.
    _bound = float(np.max(np.abs(difference)))
    _norm = plt.Normalize(-_bound, _bound)
    _fig, _ax = export.bare_axes(bounds)
    for _, _rows, _corners in export.cell_polygons(cases["planar", "specular", finest].mesh):
        _patches = PolyCollection(list(_corners), cmap=colours.DIVERGING, norm=_norm, linewidths=0.2, edgecolors="face")
        _patches.set_array(difference[_rows])
        _ax.add_collection(_patches)
    export.save_image(DATA / "difference", _fig, bounds, colourbar=export.colourbar(-_bound, _bound, colours.DIVERGING))

    mo.vstack(
        [
            mo.md(f"""
    | | |
    |---|---|
    | max absolute difference | {np.max(np.abs(difference)):.4f} MW/m^2 |
    | mean absolute difference | {np.mean(np.abs(difference)):.4f} MW/m^2 |
    | max relative to 4 sigma T^4 | {np.max(np.abs(difference)) / (4 * EMISSIVE_POWER * SCALE):.2%} |
    """),
            mo.image(figure(FIGURES, "difference")),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
