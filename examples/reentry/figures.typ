// Figures of the re-entry shock layer, drawn from reentry.py's data/.

#import "../style.typ": *

#let panel-height = 7cm

#let load(stem) = (
  png: read("data/" + stem + ".png", encoding: none),
  meta: json("data/" + stem + ".json"),
)

#let axes = (xlabel: [$x$ [#quan[m]]], ylabel: [$r$ [#quan[m]]])

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
    colourbar(whole.meta.colourbar, label: [Volumetric heating [#quan[GW/m^3]]], height: panel-height),
  )
}

// Radiative flux along the capsule surface, nose to base.
#let wall-flux() = {
  // Both flux columns are in MW/m^2.
  let faces = lq.load-txt(read("data/wall_flux.csv"), header: true)
  let bins = lq.load-txt(read("data/wall_flux_binned.csv"), header: true)
  let shoulder = json("data/wall_flux.json").shoulder
  // A step line holds each bin's value from its low edge to the next, so the
  // last value repeats at the high edge of the last bin.
  let step-x = bins.lo + (bins.hi.last(),)
  let step-y = bins.flux + (bins.flux.last(),)
  // Framed on the pooled curve: the per-face points run off the top near the
  // nose, where the rings they revolve into carry no area.
  let ceiling = 1.2 * calc.max(..bins.flux)
  lq.diagram(
    width: 10cm,
    height: 6cm,
    xlabel: [Surface distance from the stagnation point [#quan[m]]],
    ylabel: [Radiative wall flux [#quan[MW/m^2]]],
    ylim: (0, ceiling),
    legend: (position: left + bottom),
    lq.scatter(
      faces.s,
      faces.flux,
      size: 2pt,
      color: palette.grey.lighten(40%),
      stroke: none,
      clip: true,
      label: [per face],
    ),
    lq.plot(
      step-x,
      step-y,
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
