// Figures of the re-entry shock layer, drawn from reentry.py's data/.

#import "../style.typ": *

#let panel-height = 7cm

#let axes = (xlabel: [$x$ [#quan[m]]], ylabel: [$r$ [#quan[m]]])

// The mesh, whole and zoomed onto the forebody, coloured by element type.
#let mesh() = {
  let whole = (png: read("data/mesh_whole.png", encoding: none), meta: json("data/mesh_whole.json"))
  let zoom = (png: read("data/mesh_zoom.png", encoding: none), meta: json("data/mesh_zoom.json"))
  grid(
    columns: 2,
    column-gutter: 1em,
    lq.diagram(
      ..extent(whole.meta),
      width: auto,
      height: panel-height,
      ..axes,
      legend: (position: left + top),
      field(whole.png, whole.meta),
      ..swatches(whole.meta.legend),
    ),
    lq.diagram(..extent(zoom.meta), width: auto, height: panel-height, ..axes, field(zoom.png, zoom.meta)),
  )
}

// Volumetric heating, whole and zoomed onto the forebody.
#let heating() = {
  let whole = (png: read("data/heating_whole.png", encoding: none), meta: json("data/heating_whole.json"))
  let zoom = (png: read("data/heating_zoom.png", encoding: none), meta: json("data/heating_zoom.json"))
  // Both panels share one colour range, so one bar serves both.
  let whole-mesh = field(whole.png, whole.meta)
  grid(
    columns: 3,
    column-gutter: 1em,
    lq.diagram(..extent(whole.meta), width: auto, height: panel-height, ..axes, whole-mesh),
    lq.diagram(..extent(zoom.meta), width: auto, height: panel-height, ..axes, field(zoom.png, zoom.meta)),
    lq.colorbar(whole-mesh, label: [Volumetric heating [#quan[GW/m^3]]], height: panel-height),
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
      step: end,
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
