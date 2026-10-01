import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import PolyCollection

    from raytrax import interactions, sources, transport
    from raytrax import scene as scene_module
    from raytrax.interactions import Kind
    from raytrax.io import vtk
    from raytrax.mesh import geometry
    from raytrax.paths import AxisymmetricPath
    from raytrax.problem import Problem
    from raytrax_examples import cases, colours, export, meshes
    from raytrax_examples.typst import figure

    HERE = Path(__file__).parent
    DATA = HERE / "data"
    FIGURES = HERE / "figures.typ"

    STEFAN_BOLTZMANN = 5.670374419e-8

    # The forebody, on the scale of the stagnation standoff: (xmin, ymin, xmax, ymax).
    ZOOM_BOUNDS = (-0.04, -0.005, 0.14, 0.26)
    # Bins of equal arc length along the wall, for the pooled wall flux.
    WALL_BINS = 40

    # The capsule is solid and unmeshed, so it is drawn as a grey silhouette.
    CAPSULE = "#cfd2d4"
    OUTLINE = "#3f3f3f"
    CELL_EDGE = "#26262699"


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Radiation in a re-entry shock layer

    The domain is the shock layer around a Stardust-like capsule: a 60 degree
    sphere-cone forebody, a shoulder, and a conical backshell, meshed as the
    `(x, r)` half-plane and revolved about `r = 0`. The capsule is solid, so it
    is not meshed at all -- it is the `wall` boundary, and the grey silhouette
    in every figure below.

    Four boundaries close the domain:

    - **shock** - the bow shock, treated as a black surface at `T`. It radiates
      `sigma T^4` per unit area into the layer and absorbs whatever comes back
      out.
    - **wall** - the capsule surface, cold and black. What lands here is the
      radiative load on the heat shield, and it is the answer this notebook
      exists to produce.
    - **outflow** - the plane through the base of the capsule. Gas escapes
      there, and so does radiation.
    - **axis** - the short stretch of `r = 0` between the shock and the nose.
      Nothing should ever reach it, and the energy table checks that.

    The gas is cold and grey: it absorbs at `kappa` and emits nothing, so every
    photon here started at the shock. There is no closed form to check against,
    only the energy budget, which has to balance exactly.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Where the geometry comes from

    Nothing behind this is a CFD solution. The capsule carries Stardust's
    published dimensions -- 0.2202 m nose radius, a 60 degree sphere-cone
    forebody, 0.8128 m base diameter, a 0.0202 m shoulder -- and the backshell
    half-angle is then whatever lands the base where the published 0.5 m
    overall length puts it. The 10,000 K below is a round number, not a
    computed post-shock temperature.

    The shock is Billig's hyperbola. Its vertex sits one standoff, 0.02 m,
    ahead of the nose; it has 1.143 nose radii of curvature there, which is
    Billig's high-Mach limit; and it runs out to an asymptote at `shock_angle`
    to the axis. Held at the cone angle, as it is here, the far shock runs
    parallel to the forebody, so the layer thickens steadily from the
    stagnation standoff instead of pinching shut against the shoulder.

    The price of that is the domain aft of the shoulder. The shock keeps
    sweeping out at 60 degrees, so by the base plane it stands two and a half
    base radii off the axis, and most of the cells end up in an expansion
    region carrying no gradient. That is the real shape, and it is why every
    field below is drawn twice: once whole, once zoomed onto the forebody, on
    the scale of the standoff itself.

    ## What the mesh is

    Graded by distance from the stagnation point and recombined to quads: about
    ten cells across the standoff at the nose, twenty times coarser out in the
    expansion, and no clustering at the wall, because an inviscid finite-volume
    solve has no boundary layer to resolve. Triangles survive only where the
    layer is too narrow to pair cells off, which is what the element counts in
    the mesh figure show.

    `reentry()` in `raytrax_examples/meshes.py` builds it, and every dimension
    named above is a keyword argument there.
    """)
    return


@app.cell
def _():
    temperature = mo.ui.slider(5000, 15000, 500, value=10000, label="Shock temperature (K)")
    kappa = mo.ui.slider(0.1, 20.0, 0.1, value=2.0, label="Absorptivity kappa (1/m)")
    num_rays = mo.ui.dropdown(["200000", "1000000", "4000000"], value="1000000", label="Rays")
    mo.hstack([temperature, kappa, num_rays], justify="start", gap=2)
    return kappa, num_rays, temperature


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## The problem

    The mesh is the `(x, r)` half-plane. Revolving it turns each cell into the
    ring it sweeps about the axis, which is the volume the axisymmetric path
    deposits into.
    """)
    return


@app.cell
def _():
    mesh = vtk.read(meshes.path("reentry"))
    geom = geometry.revolve(mesh.geometry)
    scene = scene_module.build(mesh.topology, geom, mesh.groups)
    scene_module.validate(mesh.topology, geom)

    shock = cases.group_faces(mesh, "shock")
    wall = cases.group_faces(mesh, "wall")
    axis = cases.group_faces(mesh, "axis")

    # Nothing here is a mirror, so a ray is finished the first time it reaches
    # any boundary. ABSORB rather than ESCAPE on the shock and the wall, so that
    # their deposits can be read back as a flux.
    table = interactions.build(
        mesh.groups,
        {"shock": Kind.ABSORB, "wall": Kind.ABSORB, "outflow": Kind.ESCAPE, "axis": Kind.ABSORB},
    )
    return axis, geom, mesh, scene, shock, table, wall


@app.cell
def _(geom, kappa, mesh, scene, shock, table, temperature):
    # The shock radiates sigma T^4 per unit area. Revolved, `faces.area` is
    # already the area of the ring a face sweeps.
    source = sources.build(
        cell_power=jnp.zeros(mesh.topology.num_cells),
        face_power=STEFAN_BOLTZMANN * temperature.value**4 * geom.faces.area[shock],
        face=shock,
    )
    problem = Problem(
        scene,
        path=AxisymmetricPath(),
        interactions=table,
        absorptivity=jnp.full(mesh.topology.num_cells, kappa.value),
        sources=source,
    )
    return (problem,)


@app.cell
def _(num_rays, problem):
    tally = transport.trace(jax.random.key(4), problem, num_rays=int(num_rays.value), max_steps=2000)
    deposit = tally.estimate(problem)
    return deposit, tally


@app.cell
def _(deposit, geom):
    # Energy per unit volume of each cell, and per unit area of each face. A
    # face on the axis sweeps no ring at all, so it can never be reached.
    heating = np.asarray(deposit.cells) / geom.cells.volume
    face_deposit = np.asarray(deposit.faces)
    flux = face_deposit / np.where(geom.faces.area > 0, geom.faces.area, np.inf)
    return face_deposit, flux, heating


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## The energy budget

    Every joule emitted at the shock has to end up somewhere. A ray reaches
    `r = 0` only with no angular momentum at all, so anything on the axis is a
    leak through the symmetry axis and must be zero.
    """)
    return


@app.cell
def _(axis, deposit, face_deposit, problem, shock, tally, wall):
    emitted = float(problem.sources.total)
    budget = {
        "absorbed in the gas": float(deposit.cells.sum()),
        "absorbed at the wall": float(face_deposit[wall].sum()),
        "back out through the shock": float(face_deposit[shock].sum()),
        "out through the outflow": float(deposit.escaped),
        "through the axis (must be zero)": float(face_deposit[axis].sum()),
        # A lane the step cap caught part-way still holds energy.
        "still in flight": float(tally.in_flight(problem)),
    }
    accounted = sum(budget.values())

    mo.md(
        "| energy | MW | fraction |\n|---|---|---|\n"
        + f"| emitted at the shock | {emitted / 1e6:,.1f} | 1.0000 |\n"
        + "".join(f"| {name} | {value / 1e6:,.1f} | {value / emitted:.4f} |\n" for name, value in budget.items())
        + f"| **accounted** | **{accounted / 1e6:,.1f}** | error {abs(accounted / emitted - 1):.2e} |"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## The mesh

    Each panel of a field figure leaves this notebook as a bare image of the
    cells, cropped to its bounds, with a JSON sidecar that tells Typst those
    bounds and what the colours mean. Typst draws the axes, the legend and the
    colour bar around it.
    """)
    return


@app.cell
def _(mesh, wall):
    # The capsule outline, from the wall faces. The profile runs strictly
    # forward in x from nose to base, so sorting the corners on x walks it in
    # order. Closing it along the axis gives a polygon to fill.
    wall_corners = mesh.geometry.vertices[mesh.topology.face_vertices[wall][:, :2]].reshape(-1, 2)
    wall_corners = wall_corners[np.argsort(wall_corners[:, 0])]
    capsule = np.vstack([wall_corners, [[wall_corners[-1, 0], 0.0]], [[wall_corners[0, 0], 0.0]]])

    # The whole domain: every cell, and the capsule.
    _points = np.vstack([capsule, mesh.geometry.vertices])
    whole_bounds = (*_points.min(axis=0).tolist(), *_points.max(axis=0).tolist())
    return capsule, whole_bounds


@app.cell
def _(capsule, mesh, whole_bounds):
    # One flat colour per element type, from lilaq's default cycle.
    mesh_legend = [
        {"label": element.name, "count": len(rows), "colour": colour, "alpha": 0.55}
        for (element, rows, _), colour in zip(export.cell_polygons(mesh), colours.CYCLE, strict=False)
    ]
    for _name, _bounds in (("mesh_whole", whole_bounds), ("mesh_zoom", ZOOM_BOUNDS)):
        _fig, _ax = export.bare_axes(_bounds)
        for (_, _, _corners), _entry in zip(export.cell_polygons(mesh), mesh_legend, strict=False):
            _ax.add_collection(
                PolyCollection(
                    list(_corners), facecolors=_entry["colour"], edgecolors=CELL_EDGE, linewidths=0.25, alpha=0.55
                )
            )
        _ax.fill(capsule[:, 0], capsule[:, 1], color=CAPSULE, ec=OUTLINE, lw=0.8, zorder=3)
        export.save_image(DATA / _name, _fig, _bounds, legend=mesh_legend)

    mo.image(figure(FIGURES, "mesh"))
    return


@app.cell
def _(capsule, heating, mesh, whole_bounds):
    heating_gw = heating / 1e9
    _norm = plt.Normalize(heating_gw.min(), heating_gw.max())
    _bar = export.colourbar(heating_gw.min(), heating_gw.max(), colours.SEQUENTIAL)
    for _name, _bounds in (("heating_whole", whole_bounds), ("heating_zoom", ZOOM_BOUNDS)):
        _fig, _ax = export.bare_axes(_bounds)
        for _, _rows, _corners in export.cell_polygons(mesh):
            # The edge colour follows the face: at zero width the gaps between
            # cells still show through as white hairlines.
            _cells = PolyCollection(
                list(_corners), cmap=colours.SEQUENTIAL, norm=_norm, linewidths=0.2, edgecolors="face"
            )
            _cells.set_array(heating_gw[_rows])
            _ax.add_collection(_cells)
        _ax.fill(capsule[:, 0], capsule[:, 1], color=CAPSULE, ec=OUTLINE, lw=0.8, zorder=3)
        export.save_image(DATA / _name, _fig, _bounds, colourbar=_bar)

    mo.image(figure(FIGURES, "heating"))
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## The radiative load on the wall
    """)
    return


@app.cell
def _(face_deposit, flux, geom, mesh, wall):
    # The wall faces from the stagnation point back, and the distance along the
    # surface to the middle of each. In (x, r) a face is a segment, and its
    # "area" is its length.
    wall_faces = wall[np.argsort(mesh.geometry.faces.centroid[wall, 0])]
    face_length = mesh.geometry.faces.area[wall_faces]
    surface_distance = np.cumsum(face_length) - face_length / 2
    shoulder = surface_distance[np.argmax(mesh.geometry.faces.centroid[wall_faces, 1])]

    # Pool each bin: the energy its faces caught over the area they revolve into.
    bin_edges = np.linspace(0.0, surface_distance[-1] + face_length[-1] / 2, WALL_BINS + 1)
    bin_index = np.clip(np.digitize(surface_distance, bin_edges) - 1, 0, WALL_BINS - 1)
    bin_energy = np.bincount(bin_index, face_deposit[wall_faces], WALL_BINS)
    bin_area = np.bincount(bin_index, geom.faces.area[wall_faces], WALL_BINS)
    pooled_flux = bin_energy / bin_area

    # Written in MW/m^2, the unit the figure shows.
    export.save_csv(DATA / "wall_flux.csv", (surface_distance, flux[wall_faces] / 1e6), "s,flux")
    export.save_csv(DATA / "wall_flux_binned.csv", (bin_edges[:-1], bin_edges[1:], pooled_flux / 1e6), "lo,hi,flux")
    export.save_json(DATA / "wall_flux.json", {"shoulder": float(shoulder)})

    mo.image(figure(FIGURES, "wall-flux"))
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    The forebody load is nearly flat and then falls off a cliff at the
    shoulder: past it the wall turns away from the shock and sees much less of
    it.

    The grey scatter near the nose is sampling, not physics. A face there
    revolves into a ring of almost no area, so hardly any rays land on it and
    its own flux is one or two deposits over almost nothing. The blue curve is
    the estimator to read.

    ### The binning

    Forty bins of equal *arc length* along the wetted profile, measured from
    the stagnation point, and each bin's flux is the energy its faces caught
    divided by the area those faces revolve into. Three choices in that, each
    of which could have gone the other way:

    - **Pooled, not averaged.** The bin value is `sum(energy) / sum(area)`,
      not the mean of the per-face fluxes it contains. Averaging the fluxes
      would weight a hairline ring at the nose exactly as heavily as a broad
      one further out, which is what puts the grey points where they are. This
      way a face contributes in proportion to the area it actually presents,
      and the bin means what it says: the mean flux over that strip of wall.
    - **Arc length, not area.** Equal-area bins would crowd the shoulder,
      where the rings are widest, and swallow the whole nose in one. Equal arc
      keeps the resolution uniform along the surface, which is what it takes
      to read *where* the cliff at the shoulder falls.
    - **Forty of them.** Fine enough that the drop at the shoulder lands in a
      single bin rather than being smeared across several, and coarse enough
      that the first bin, always the worst-sampled one, still gathers a few
      hundred rays at a million launched. `WALL_BINS` is the knob.
      The plateau sits at the same level anywhere from twenty bins to a
      hundred; past that the first bin is down to tens of rays, and since
      there are only a couple of hundred wall faces in the whole profile,
      bins start coming out holding no face at all.
    """)
    return


if __name__ == "__main__":
    app.run()
