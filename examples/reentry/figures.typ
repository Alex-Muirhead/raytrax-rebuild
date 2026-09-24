// Figures of the re-entry shock layer, drawn from reentry.py's data/.

#import "../style.typ": *

#let panel-height = 7cm

#let load(stem) = (
  png: read("data/" + stem + ".png", encoding: none),
  meta: json("data/" + stem + ".json"),
)

#let axes = (xlabel: labelled($x$, "m"), ylabel: labelled($r$, "m"))

// The mesh, whole and zoomed onto the forebody, coloured by element type.
#let mesh() = {
  let whole = load("mesh_whole")
  let zoom = load("mesh_zoom")
  grid(
    columns: 2,
    column-gutter: 1em,
    equal-diagram(
      whole.meta,
      height: panel-height,
      ..axes,
      legend: (position: left + top),
      field(whole.png, whole.meta),
      ..swatches(whole.meta.legend),
    ),
    equal-diagram(zoom.meta, height: panel-height, ..axes, field(zoom.png, zoom.meta)),
  )
}

// Volumetric heating, whole and zoomed onto the forebody.
#let heating() = {
  let whole = load("heating_whole")
  let zoom = load("heating_zoom")
  grid(
    columns: 3,
    column-gutter: 1em,
    equal-diagram(whole.meta, height: panel-height, ..axes, field(whole.png, whole.meta)),
    equal-diagram(zoom.meta, height: panel-height, ..axes, field(zoom.png, zoom.meta)),
    colourbar(whole.meta.colourbar, height: panel-height),
  )
}

// Radiative flux along the capsule surface, nose to base.
#let wall-flux() = {
  let faces = records(csv("data/wall_flux.csv"))
  let bins = records(csv("data/wall_flux_binned.csv")).filter(row => not row.flux.is-nan())
  let shoulder = json("data/wall_flux.json").shoulder
  let pooled = column(bins, "flux").map(flux => flux / 1e6)
  // Framed on the pooled curve: the per-face points run off the top near the
  // nose, where the rings they revolve into carry no area.
  let ceiling = 1.2 * calc.max(..pooled)
  lq.diagram(
    width: 10cm,
    height: 6cm,
    xlabel: labelled([Surface distance from the stagnation point], "m"),
    ylabel: labelled([Radiative wall flux], "MW/m^2"),
    ylim: (0, ceiling),
    legend: (position: left + bottom),
    lq.scatter(
      column(faces, "s"),
      column(faces, "flux").map(flux => flux / 1e6),
      size: 2pt,
      color: palette.grey.lighten(40%),
      stroke: none,
      clip: true,
      label: [per face],
    ),
    lq.plot(
      column(bins, "lo") + (bins.last().hi,),
      pooled + (pooled.last(),),
      step: start,
      mark: none,
      stroke: 1.2pt + palette.blue,
      label: [pooled over the bin],
    ),
    lq.vlines(shoulder, stroke: (paint: palette.grey, dash: "dotted")),
    lq.place(shoulder, ceiling, align: right + top, pad(right: 3pt, top: 3pt, text(size: 0.8em)[shoulder])),
  )
}

#let figures = (mesh: mesh, heating: heating, wall-flux: wall-flux)

#standalone(figures)
