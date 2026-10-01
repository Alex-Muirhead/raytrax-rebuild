// Shared figure style for the examples.
//
// Every example's figures.typ imports this, defines one function per figure,
// and ends in `#standalone(...)`. The thesis imports a figures.typ and calls
// its functions; figures inherit the text style of the document they sit in.
//
// Paths in Typst resolve against the file that names them, so a figures.typ
// reads its own data and hands the bytes and dictionaries to these helpers.

#import "@preview/lilaq:0.6.0" as lq
#import "@preview/zero:0.7.1": num, quan, set-group, set-num, set-unit

#let number-setup = {
  set-num(product: sym.times)
  set-group(threshold: 5)
  set-unit(fraction: "power")
}

// lilaq's default line colours. Python fills mesh element types from the same
// map, `colours.CYCLE` in raytrax_examples.
#let cycle = lq.color.map.petroff10
#let palette = (blue: cycle.at(0), orange: cycle.at(1), grey: cycle.at(8))

// A colour bar for a bare image, from its sidecar `colourbar` entry, which
// names the lilaq colour map the image was drawn with.
#let colourbar(bar, ..args) = {
  let mesh = lq.colormesh(
    (0, 1),
    (0, 1),
    ((bar.min, bar.max),),
    map: dictionary(lq.color.map).at(bar.colourmap),
    min: bar.min,
    max: bar.max,
  )
  lq.colorbar(mesh, ..args)
}

// A bare image stretched over the data bounds its sidecar gives. lilaq's default
// alignment spans the rect from its origin towards larger y.
#let field(png, meta) = lq.rect(
  meta.xmin,
  meta.ymin,
  width: meta.xmax - meta.xmin,
  height: meta.ymax - meta.ymin,
  stroke: none,
  inset: 0pt,
  clip: true,
  z-index: 1,
  image(png, width: 100%, height: 100%, fit: "stretch"),
)

// Legend entries as filled swatches, for a sidecar `legend` list. An entry may
// give the `alpha` its fill was drawn with, and the `count` of cells it labels.
#let swatches(entries) = entries.map(entry => lq.rect(
  0,
  0,
  width: 0,
  height: 0,
  fill: rgb(entry.colour).transparentize((1 - entry.at("alpha", default: 1)) * 100%),
  stroke: none,
  label: if "count" in entry { [#num(entry.count) #entry.label] } else { entry.label },
))

// A diagram with equal x and y scales, framed on `bounds`. Give one of `width`
// or `height`; the other follows from the aspect ratio of `bounds`.
#let equal-diagram(bounds, width: auto, height: auto, ..args) = {
  let aspect = (bounds.ymax - bounds.ymin) / (bounds.xmax - bounds.xmin)
  if width == auto and height == auto { width = 8cm }
  if width == auto { width = height / aspect } else { height = width * aspect }
  lq.diagram(
    width: width,
    height: height,
    xlim: (bounds.xmin, bounds.xmax),
    ylim: (bounds.ymin, bounds.ymax),
    ..args,
  )
}

// Rows of a CSV with a header, as dictionaries of floats.
#let records(rows) = {
  let (header, ..body) = rows
  body.map(row => header.zip(row.map(float)).to-dict())
}

// One column of the dictionaries `records` gives.
#let column(rows, name) = rows.map(row => row.at(name))

// A step line over bins, from their edges and one value per bin.
#let stairs(lo, hi, values, ..args) = lq.plot(
  lo + (hi.last(),),
  values + (values.last(),),
  step: start,
  mark: none,
  ..args,
)

// Publish the figure names, and render the one named by `--input figure=<name>`
// on a page fitted to it. An importing document sees neither.
#let standalone(figures) = {
  [#metadata(figures.keys()) <figures>]
  let name = sys.inputs.at("figure", default: none)
  set page(width: auto, height: auto, margin: 6pt)
  number-setup
  if name != none {
    figures.at(name)()
  } else {
    figures.pairs().map(((k,f)) => heading(raw(k), level: 3) + f()).join()
  }
}
