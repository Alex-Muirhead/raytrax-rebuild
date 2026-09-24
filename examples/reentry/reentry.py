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
    from raytrax_examples import cases, colours, export
    from raytrax_examples.typst import compile_figures

    HERE = Path(__file__).parent
    DATA = HERE / "data"

    STEFAN_BOLTZMANN = 5.670374419e-8

    # The capsule is solid and unmeshed, so it is drawn as a silhouette.
    CAPSULE = "#cfd2d4"
    # The forebody, on the scale of the stagnation standoff.
    ZOOM_BOUNDS = (-0.04, -0.005, 0.14, 0.26)


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


@app.function
def solve(temperature, absorptivity, rays):
    """Trace the shock layer, returning the case and its tally."""
    case = cases.prepare("reentry", absorptivity, emissive_power=STEFAN_BOLTZMANN * temperature**4)
    tally = transport.trace(jax.random.key(4), case.problem, num_rays=rays, max_steps=2000)
    return case, tally


@app.cell
def _(kappa, num_rays, temperature):
    case, tally = solve(temperature.value, kappa.value, int(num_rays.value))
    mesh, geom, problem = case.mesh, case.geom, case.problem
    deposit = tally.estimate(problem)
    heating = np.asarray(deposit.cells) / geom.cells.volume
    # A face lying on the axis sweeps no ring at all, so it can never be reached.
    flux = np.asarray(deposit.faces) / np.where(geom.faces.area > 0, geom.faces.area, np.inf)
    return deposit, flux, geom, heating, mesh, problem, tally


@app.cell
def _(deposit, geom, mesh):
    def capsule():
        """The capsule outline, closed along the axis, taken from the wall faces alone."""
        wall = cases.group_faces(mesh, "wall")
        corners = mesh.geometry.vertices[mesh.topology.face_vertices[wall][:, :2]].reshape(-1, 2)
        # The profile runs strictly forward in x from nose to base, so sorting
        # on x walks it in order and no face-to-face chasing is needed.
        profile = corners[np.argsort(corners[:, 0])]
        return np.vstack([profile, [[profile[-1, 0], 0.0]], [[profile[0, 0], 0.0]]])

    def wall_profile():
        """Wall faces from the stagnation point back, with the surface distance to each."""
        wall = cases.group_faces(mesh, "wall")
        wall = wall[np.argsort(mesh.geometry.faces.centroid[wall, 0])]
        # In (x, r), a face is a segment and its "area" is its length.
        length = mesh.geometry.faces.area[wall]
        return wall, np.cumsum(length) - length / 2, length

    def wall_flux(bins=40):
        """Wall flux against surface distance, pooled over equal-arc bins.

        Pooling deposit and area rather than averaging per-face fluxes is what
        makes the nose usable: a face there revolves into a ring of almost no
        area, so on its own it is one or two rays over almost nothing.
        """
        wall, distance, length = wall_profile()
        edges = np.linspace(0.0, distance[-1] + length[-1] / 2, bins + 1)
        index = np.clip(np.digitize(distance, edges) - 1, 0, bins - 1)
        energy = np.bincount(index, np.asarray(deposit.faces)[wall], bins)
        area = np.bincount(index, geom.faces.area[wall], bins)
        return edges, np.divide(energy, area, out=np.full(bins, np.nan), where=area > 0)

    return capsule, wall_flux, wall_profile


@app.cell(hide_code=True)
def _(geom, heating, mesh, problem):
    def _stats():
        counts: dict[str, int] = {}
        for index, element in enumerate(mesh.topology.block_elements):
            counts[element.name] = counts.get(element.name, 0) + int((mesh.topology.cell_block == index).sum())
        measure = mesh.geometry.cells.volume
        return mo.md(f"""
        | | |
        |---|---|
        | cells | {mesh.topology.num_cells:,} - {", ".join(f"{n:,} {t}" for t, n in counts.items())} |
        | boundary groups | {", ".join(mesh.groups.names[1:])} |
        | cell size spread | {np.sqrt(measure.max() / measure.min()):.1f}x |
        | shock area | {geom.faces.area[cases.group_faces(mesh, "shock")].sum():.4f} m2 |
        | wetted wall area | {geom.faces.area[cases.group_faces(mesh, "wall")].sum():.4f} m2 |
        | emitted | {float(problem.sources.total) / 1e6:,.1f} MW |
        | peak heating | {heating.max() / 1e9:.3f} GW/m3 |
        """)

    _stats()
    return


@app.cell
def _(capsule, mesh):
    def domain_bounds():
        """Bounding box of the cells and the capsule outline, for viewbox alignment."""
        points = np.vstack([capsule(), mesh.geometry.vertices])
        (xmin, ymin), (xmax, ymax) = points.min(axis=0), points.max(axis=0)
        return float(xmin), float(ymin), float(xmax), float(ymax)

    def silhouette(ax):
        """Lay the capsule over the cells."""
        outline = capsule()
        ax.fill(outline[:, 0], outline[:, 1], color=CAPSULE, zorder=3, ec="#3f3f3f", lw=0.8)

    def export_mesh(bounds):
        """The mesh cells and capsule silhouette alone, with the legend contents as data."""
        fig, ax = export.bare_axes(bounds)
        legend = []
        for (element, rows, corners), colour in zip(export.cell_polygons(mesh), colours.cycle(), strict=False):
            ax.add_collection(
                PolyCollection(list(corners), facecolors=colour, edgecolors="#26262699", linewidths=0.25, alpha=0.55)
            )
            legend.append({"label": element.name, "count": len(rows), "colour": colour, "alpha": 0.55})
        silhouette(ax)
        return fig, legend

    def export_field(bounds, values, norm, cmap=colours.SEQUENTIAL):
        """The field fill and capsule silhouette alone, no colour bar."""
        fig, ax = export.bare_axes(bounds)
        for _, rows, corners in export.cell_polygons(mesh):
            # Edge colour follows the face: at linewidth zero the gaps between
            # cells still show through as white hairlines.
            polys = PolyCollection(list(corners), cmap=cmap, norm=norm, linewidths=0.2, edgecolors="face")
            polys.set_array(values[rows])
            ax.add_collection(polys)
        silhouette(ax)
        return fig

    return domain_bounds, export_field, export_mesh


@app.cell(hide_code=True)
def _(domain_bounds, export_field, export_mesh, flux, heating, mesh, wall_flux, wall_profile):
    def _export():
        written = []
        low, high = float(heating.min() / 1e9), float(heating.max() / 1e9)
        norm = plt.Normalize(low, high)
        bar = export.colourbar(low, high, colours.SEQUENTIAL, "Volumetric heating", "GW/m^3")
        for suffix, bounds in (("whole", domain_bounds()), ("zoom", ZOOM_BOUNDS)):
            fig, legend = export_mesh(bounds)
            written += export.save_image(DATA / f"mesh_{suffix}", fig, bounds, legend=legend)
            fig = export_field(bounds, heating / 1e9, norm)
            written += export.save_image(DATA / f"heating_{suffix}", fig, bounds, colourbar=bar)

        wall, distance, _ = wall_profile()
        centroid = mesh.geometry.faces.centroid[wall]
        written.append(
            export.save_csv(
                DATA / "wall_flux.csv", (distance, centroid[:, 0], centroid[:, 1], flux[wall]), "s,x,r,flux"
            )
        )
        edges, binned = wall_flux()
        written.append(export.save_csv(DATA / "wall_flux_binned.csv", (edges[:-1], edges[1:], binned), "lo,hi,flux"))
        shoulder = distance[np.argmax(centroid[:, 1])]
        written.append(export.save_json(DATA / "wall_flux.json", {"shoulder": float(shoulder)}))

        cells = mesh.geometry.cells.centroid
        written.append(export.save_csv(DATA / "heating.csv", (cells[:, 0], cells[:, 1], heating), "x,r,heating"))
        return written + compile_figures(HERE / "figures.typ")

    written = _export()
    return (written,)


@app.cell(hide_code=True)
def _(written):
    mo.vstack([mo.image(path) for path in written if path.parent.name == "figures"])
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
      hundred rays at a million launched. `wall_flux(bins=...)` is the knob.
      The plateau sits at the same level anywhere from twenty bins to a
      hundred; past that the first bin is down to tens of rays, and since
      there are only a couple of hundred wall faces in the whole profile,
      bins start coming out holding no face at all.
    """)
    return


@app.cell(hide_code=True)
def _(deposit, flux, mesh, num_rays, problem, tally):
    def _budget():
        emitted = float(problem.sources.total)
        # A terminating ray is emptied into the deposit, so a lane still holding
        # anything is one the step cap caught part-way.
        escaped = float(deposit.escaped)
        in_flight = float(tally.in_flight(problem))
        faces = np.asarray(deposit.faces)
        gas = float(deposit.cells.sum())
        total = gas + float(deposit.faces.sum()) + escaped + in_flight
        wall = cases.group_faces(mesh, "wall")

        def row(name, value):
            return f"| {name} | {value / 1e6:,.1f} MW | {value / emitted:.4f} |"

        # A ray only reaches r = 0 with no angular momentum at all, so anything landing on
        # the axis group is a leak through the symmetry axis.
        return mo.md(f"""
        | energy | | fraction |
        |---|---|---|
        | emitted at the shock | {emitted / 1e6:,.1f} MW | 1.0000 |
        {row("absorbed in the gas", gas)}
        {row("absorbed at the wall", faces[wall].sum())}
        {row("back out through the shock", faces[cases.group_faces(mesh, "shock")].sum())}
        {row("out through the outflow", escaped)}
        {row("through the axis (must be zero)", faces[cases.group_faces(mesh, "axis")].sum())}
        {row("still in flight", in_flight)}
        | **accounted** | **{total / 1e6:,.1f} MW** | error {abs(total / emitted - 1):.2e} |

        Peak wall flux {flux[wall].max() / 1e6:,.1f} MW/m2, with {int(tally.rays.alive.sum()):,} of
        {int(num_rays.value):,} rays left alive at the step limit.
        """)

    _budget()
    return


if __name__ == "__main__":
    app.run()
