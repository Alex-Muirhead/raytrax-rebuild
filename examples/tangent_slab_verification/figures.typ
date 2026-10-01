// Figures of the tangent-slab verification, drawn from tangent_slab_verification.py's data/.

#import "../style.typ": *

#let flat = ("slab_planar", "slab_axisymmetric")
#let solid = ("cylinder_coarse", "cylinder")
// Cases with an axis, so a radial profile.
#let radial-cases = ("slab_axisymmetric", "cylinder_coarse", "cylinder")

#let load(stem) = (
  png: read("data/" + stem + ".png", encoding: none),
  meta: json("data/" + stem + ".json"),
)

#let heating-label = [Volumetric heating [#quan[/m]]]

#let flat-axes(name) = (
  xlabel: $x$,
  ylabel: if name == "slab_axisymmetric" { $r$ } else { $y$ },
)

// The mesh of one case, coloured by element type where there is more than one.
#let mesh(name) = {
  let data = load(name + "/mesh")
  if name in flat {
    equal-diagram(
      data.meta,
      width: 7cm,
      ..flat-axes(name),
      legend: (position: right + top),
      field(data.png, data.meta),
      ..swatches(data.meta.legend),
    )
  } else {
    image(data.png, width: 7cm)
  }
}

// A cell field of one case, with its colour bar.
#let cell-field(stem, name) = {
  let data = load(stem)
  let plot = if name in flat {
    equal-diagram(data.meta, width: 7cm, ..flat-axes(name), field(data.png, data.meta))
  } else {
    image(data.png, width: 7cm)
  }
  let bar = data.meta.colourbar
  let label = if bar.unit == none { bar.label } else [#bar.label [#quan(bar.unit)]]
  grid(columns: 2, column-gutter: 1em, align: horizon, plot, colourbar(bar, label: label, height: 5cm))
}

#let heating(name) = cell-field(name + "/heating", name)

// Band edges are quarters of the slab depth.
#let band-round = (mode: "places", precision: 2)

// Heating against depth, with the tangent-slab mean over each bin.
#let axial(name) = {
  let rows = records(csv("data/" + name + "/axial.csv"))
  let (lo, hi) = (column(rows, "lo"), column(rows, "hi"))
  lq.diagram(
    width: 8cm,
    height: 5cm,
    xlabel: $x$,
    ylabel: heating-label,
    stairs(lo, hi, column(rows, "heating"), stroke: 1.2pt + palette.blue, label: [Monte-Carlo]),
    stairs(
      lo,
      hi,
      column(rows, "tangent_slab"),
      stroke: (paint: palette.orange, thickness: 1.2pt, dash: "dashed"),
      label: [Tangent slab],
    ),
  )
}

// Heating against radius in each axial band, with the uniform tangent-slab mean of the band.
#let radial(name) = {
  let rows = records(csv("data/" + name + "/radial.csv"))
  let bands = rows.map(row => row.band_lo).dedup()
  lq.diagram(
    width: 8cm,
    height: 5cm,
    xlabel: $r$,
    ylabel: heating-label,
    // Outside the data area: every band spans the full radius.
    legend: (position: left + top, dx: 100% + 0.5em),
    ..bands
      .enumerate()
      .map(((index, band)) => {
        let shells = rows.filter(row => row.band_lo == band and not row.heating.is-nan())
        let colour = cycle.at(calc.rem(index, cycle.len()))
        let first = shells.first()
        (
          stairs(
            column(shells, "lo"),
            column(shells, "hi"),
            column(shells, "heating"),
            stroke: 1pt + colour,
            label: [#num(first.band_lo, round: band-round) $< x <$ #num(first.band_hi, round: band-round)],
          ),
          lq.hlines(first.tangent_slab, stroke: (paint: colour, thickness: 0.6pt, dash: "dashed")),
        )
      })
      .flatten(),
  )
}

// The axisymmetric heating minus the planar heating, cell for cell.
#let difference() = cell-field("comparison/difference", "slab_planar")

// That difference as a fraction of the peak planar heating.
#let relative() = cell-field("comparison/relative", "slab_planar")

#let figures = {
  let figures = (:)
  for name in flat + solid {
    let key = name.replace("_", "-")
    figures.insert("mesh-" + key, () => mesh(name))
    if name != "cylinder_coarse" {
      figures.insert("heating-" + key, () => heating(name))
      figures.insert("axial-" + key, () => axial(name))
    }
    if name in radial-cases and name != "cylinder_coarse" {
      figures.insert("radial-" + key, () => radial(name))
    }
  }
  figures.insert("difference", difference)
  figures.insert("relative", relative)
  figures
}

#standalone(figures)
