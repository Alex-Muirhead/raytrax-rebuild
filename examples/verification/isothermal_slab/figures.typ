// Figures of the isothermal-slab verification, drawn from isothermal_verification.py's data/.

#import "../../style.typ": *

#let incident-label = [Incident radiation [#quan[MW/m^2]]]

// The vertical label of a reading: y extruded, r revolved.
#let vertical = (planar: $y$, axisymmetric: $r$)

// The band the strip profiles average over, outlined on a field.
#let strip-outline(meta) = lq.rect(
  meta.xmin,
  meta.strip.at(0),
  width: meta.xmax - meta.xmin,
  height: meta.strip.at(1) - meta.strip.at(0),
  fill: none,
  stroke: (paint: palette.orange, thickness: 1pt, dash: "dashed"),
)

// One structured grid, with the strip outlined.
#let mesh(cells) = {
  let png = read("data/mesh_" + str(cells) + ".png", encoding: none)
  let meta = json("data/mesh_" + str(cells) + ".json")
  lq.diagram(
    ..extent(meta),
    width: 5cm,
    height: auto,
    xlabel: $x$,
    ylabel: $y$,
    field(png, meta),
    strip-outline(meta),
  )
}

// Incident radiation on the finest grid, for one reading and one boundary kind.
#let incident(reading, boundary) = {
  let stem = "data/incident_" + reading + "_" + boundary
  let png = read(stem + ".png", encoding: none)
  let meta = json(stem + ".json")
  let colormesh = field(png, meta)
  grid(
    columns: 2,
    column-gutter: 1em,
    align: horizon,
    lq.diagram(
      ..extent(meta),
      width: 6cm,
      height: auto,
      xlabel: $x$,
      ylabel: vertical.at(reading),
      colormesh,
      strip-outline(meta),
    ),
    lq.colorbar(colormesh, label: incident-label, height: 5cm),
  )
}

// The mean of each column within the strip, one step line per grid, with error
// bars of one standard deviation down the column at the bin centres, against
// the slab solution.
#let strip(reading, boundary) = {
  let slab = lq.load-txt(read("data/slab.csv"), header: true)
  lq.diagram(
    width: 8cm,
    height: 5cm,
    xlabel: $x$,
    ylabel: incident-label,
    legend: (position: center + bottom),
    ..for (index, cells) in (15, 30, 45).enumerate() {
      let bins = lq.load-txt(
        read("data/strip_" + reading + "_" + boundary + "_" + str(cells) + ".csv"),
        header: true,
      )
      let colour = cycle.at(index)
      // A step line holds each bin's value from its low edge to the next, so
      // the last value repeats at the high edge of the last bin.
      (
        lq.plot(
          bins.lo + (bins.hi.last(),),
          bins.incident + (bins.incident.last(),),
          step: end,
          mark: none,
          stroke: 1pt + colour,
          label: [#cells #sym.times #cells],
        ),
        lq.plot(
          bins.lo.zip(bins.hi).map(((lo, hi)) => (lo + hi) / 2),
          bins.incident,
          yerr: bins.spread,
          mark: none,
          stroke: none,
          color: colour,
        ),
      )
    },
    lq.plot(
      slab.x,
      slab.incident,
      mark: none,
      stroke: (paint: palette.grey, thickness: 1pt, dash: "dashed"),
      label: [Isothermal slab],
    ),
  )
}

// The axisymmetric minus the planar incident radiation, with mirrored sides.
#let difference() = {
  let png = read("data/difference.png", encoding: none)
  let meta = json("data/difference.json")
  let colormesh = field(png, meta)
  grid(
    columns: 2,
    column-gutter: 1em,
    align: horizon,
    lq.diagram(..extent(meta), width: 6cm, height: auto, xlabel: $x$, ylabel: $y$, colormesh),
    lq.colorbar(colormesh, label: [Difference [#quan[MW/m^2]]], height: 5cm),
  )
}

#standalone((
  mesh-15: () => mesh(15),
  mesh-30: () => mesh(30),
  mesh-45: () => mesh(45),
  incident-planar-specular: () => incident("planar", "specular"),
  incident-planar-absorbing: () => incident("planar", "absorbing"),
  incident-axisymmetric-specular: () => incident("axisymmetric", "specular"),
  incident-axisymmetric-absorbing: () => incident("axisymmetric", "absorbing"),
  strip-planar-specular: () => strip("planar", "specular"),
  strip-planar-absorbing: () => strip("planar", "absorbing"),
  strip-axisymmetric-specular: () => strip("axisymmetric", "specular"),
  strip-axisymmetric-absorbing: () => strip("axisymmetric", "absorbing"),
  difference: difference,
))
