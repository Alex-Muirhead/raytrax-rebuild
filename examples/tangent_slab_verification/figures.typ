// Figures of the tangent-slab verification, drawn from tangent_slab_verification.py's data/.

#import "../style.typ": *

// The hot end emits 1 per unit area, so heating per unit volume is in 1/m.
#let heating-label = [Volumetric heating [#quan[/m]]]

// Band edges are quarters of the slab depth.
#let band-round = (mode: "places", precision: 2)

// The slab mesh, read flat, coloured by element type.
#let mesh-slab-planar() = {
  let png = read("data/slab_mesh.png", encoding: none)
  let meta = json("data/slab_mesh.json")
  lq.diagram(
    ..extent(meta),
    width: 7cm,
    height: auto,
    xlabel: $x$,
    ylabel: $y$,
    legend: (position: right + top),
    field(png, meta),
    ..swatches(meta.legend),
  )
}

// The same slab mesh, read as the (x, r) half-plane.
#let mesh-slab-axisymmetric() = {
  let png = read("data/slab_mesh.png", encoding: none)
  let meta = json("data/slab_mesh.json")
  lq.diagram(
    ..extent(meta),
    width: 7cm,
    height: auto,
    xlabel: $x$,
    ylabel: $r$,
    legend: (position: right + top),
    field(png, meta),
    ..swatches(meta.legend),
  )
}

// The coarse cylinder, cut in half.
#let mesh-cylinder-coarse() = image("data/cylinder_coarse_mesh.png", width: 7cm)

// The cylinder, cut in half.
#let mesh-cylinder() = image("data/cylinder_mesh.png", width: 7cm)

// Volumetric heating in the slab, read flat.
#let heating-slab-planar() = {
  let png = read("data/slab_planar_heating.png", encoding: none)
  let meta = json("data/slab_planar_heating.json")
  let colormesh = field(png, meta)
  grid(
    columns: 2,
    column-gutter: 1em,
    align: horizon,
    lq.diagram(..extent(meta), width: 7cm, height: auto, xlabel: $x$, ylabel: $y$, colormesh),
    lq.colorbar(colormesh, label: heating-label, height: 5cm),
  )
}

// Volumetric heating in the slab, read as the (x, r) half-plane.
#let heating-slab-axisymmetric() = {
  let png = read("data/slab_axisymmetric_heating.png", encoding: none)
  let meta = json("data/slab_axisymmetric_heating.json")
  let colormesh = field(png, meta)
  grid(
    columns: 2,
    column-gutter: 1em,
    align: horizon,
    lq.diagram(..extent(meta), width: 7cm, height: auto, xlabel: $x$, ylabel: $r$, colormesh),
    lq.colorbar(colormesh, label: heating-label, height: 5cm),
  )
}

// Volumetric heating in the cylinder, cut in half.
#let heating-cylinder() = {
  let bar = json("data/cylinder_heating.json").colourbar
  // The render is not a field over (x, y), so this colour mesh only carries the
  // range and colour map, for the colour bar.
  let colormesh = lq.colormesh(
    (0, 1),
    (0, 1),
    ((bar.min, bar.max),),
    map: dictionary(lq.color.map).at(bar.colourmap),
    min: bar.min,
    max: bar.max,
  )
  grid(
    columns: 2,
    column-gutter: 1em,
    align: horizon,
    image("data/cylinder_heating.png", width: 7cm),
    lq.colorbar(colormesh, label: heating-label, height: 5cm),
  )
}

// Heating against depth from an `axial.csv`, with the tangent-slab mean over each bin.
#let axial-heating(path) = {
  let bins = lq.load-txt(read(path), header: true)
  // A step line holds each bin's value from its low edge to the next, so the
  // last value repeats at the high edge of the last bin.
  let step-x = bins.lo + (bins.hi.last(),)
  let monte-carlo = bins.heating + (bins.heating.last(),)
  let tangent-slab = bins.tangent_slab + (bins.tangent_slab.last(),)
  lq.diagram(
    width: 8cm,
    height: 5cm,
    xlabel: $x$,
    ylabel: heating-label,
    lq.plot(step-x, monte-carlo, step: end, mark: none, stroke: 1.2pt + palette.blue, label: [Monte-Carlo]),
    lq.plot(
      step-x,
      tangent-slab,
      step: end,
      mark: none,
      stroke: (paint: palette.orange, thickness: 1.2pt, dash: "dashed"),
      label: [Tangent slab],
    ),
  )
}

#let axial-slab-planar() = axial-heating("data/slab_planar_axial.csv")
#let axial-slab-axisymmetric() = axial-heating("data/slab_axisymmetric_axial.csv")
#let axial-cylinder() = axial-heating("data/cylinder_axial.csv")

// Heating against radius in each axial band, from a `radial.csv` of shell means
// and a `radial.json` of bands, each band with its tangent-slab mean, which is
// uniform in radius.
#let radial-heating(shells-path, bands-path) = {
  let shells = lq.load-txt(read(shells-path), header: true)
  let bands = json(bands-path).bands
  let step-x = shells.lo + (shells.hi.last(),)
  lq.diagram(
    width: 8cm,
    height: 5cm,
    xlabel: $r$,
    ylabel: heating-label,
    // Outside the data area: every band spans the full radius.
    legend: (position: left + top, dx: 100% + 0.5em),
    // Each band is a column `band_1`, `band_2`, ... of the shells, and a
    // dashed line at its tangent-slab mean, in the band's own colour.
    ..for (index, band) in bands.enumerate() {
      let heating = shells.at("band_" + str(index + 1))
      let colour = cycle.at(index)
      (
        lq.plot(
          step-x,
          heating + (heating.last(),),
          step: end,
          mark: none,
          stroke: 1pt + colour,
          label: [#num(band.lo, round: band-round) $< x <$ #num(band.hi, round: band-round)],
        ),
        lq.hlines(band.tangent_slab, stroke: (paint: colour, thickness: 0.6pt, dash: "dashed")),
      )
    },
  )
}

#let radial-slab-axisymmetric() = radial-heating(
  "data/slab_axisymmetric_radial.csv",
  "data/slab_axisymmetric_radial.json",
)
#let radial-cylinder() = radial-heating("data/cylinder_radial.csv", "data/cylinder_radial.json")

// The axisymmetric heating minus the planar heating, cell for cell.
#let difference() = {
  let png = read("data/difference.png", encoding: none)
  let meta = json("data/difference.json")
  let colormesh = field(png, meta)
  grid(
    columns: 2,
    column-gutter: 1em,
    align: horizon,
    lq.diagram(..extent(meta), width: 7cm, height: auto, xlabel: $x$, ylabel: $y$, colormesh),
    lq.colorbar(colormesh, label: [Difference [#quan[/m]]], height: 5cm),
  )
}

// That difference as a fraction of the peak planar heating.
#let relative() = {
  let png = read("data/relative.png", encoding: none)
  let meta = json("data/relative.json")
  let colormesh = field(png, meta)
  grid(
    columns: 2,
    column-gutter: 1em,
    align: horizon,
    lq.diagram(..extent(meta), width: 7cm, height: auto, xlabel: $x$, ylabel: $y$, colormesh),
    lq.colorbar(colormesh, label: [Relative to peak planar heating], height: 5cm),
  )
}

#standalone((
  mesh-slab-planar: mesh-slab-planar,
  mesh-slab-axisymmetric: mesh-slab-axisymmetric,
  mesh-cylinder-coarse: mesh-cylinder-coarse,
  mesh-cylinder: mesh-cylinder,
  heating-slab-planar: heating-slab-planar,
  heating-slab-axisymmetric: heating-slab-axisymmetric,
  heating-cylinder: heating-cylinder,
  axial-slab-planar: axial-slab-planar,
  axial-slab-axisymmetric: axial-slab-axisymmetric,
  axial-cylinder: axial-cylinder,
  radial-slab-axisymmetric: radial-slab-axisymmetric,
  radial-cylinder: radial-cylinder,
  difference: difference,
  relative: relative,
))
