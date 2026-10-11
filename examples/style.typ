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
  set-unit(fraction: "inline")
}

// lilaq's default line colours. Python fills mesh element types from the same
// map, `colours.CYCLE` in raytrax_examples.
#let cycle = lq.color.map.petroff10
#let palette = (blue: cycle.at(0), orange: cycle.at(1), grey: cycle.at(8))

// A bare image placed on the bounds its sidecar gives, as a colour mesh. With a
// sidecar `colourbar` entry, the mesh carries its range and lilaq colour map,
// so `lq.colorbar` can draw it.
#let field(png, meta) = {
  let bar = meta.at("colourbar", default: none)
  let colours = if bar == none { (:) } else {
    (map: dictionary(lq.color.map).at(bar.colourmap), min: bar.min, max: bar.max)
  }
  // The top row of the image is the top of the bounds, so y runs from ymax.
  lq.colormesh((meta.xmin, meta.xmax), (meta.ymax, meta.ymin), image(png), ..colours)
}

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

// The `lq.diagram` arguments that frame a bare image on the bounds its sidecar
// gives, with equal x and y scales. Set one of `width` or `height` to `auto`
// at the call, and lilaq sizes it from the other.
#let extent(meta) = (
  xlim: (meta.xmin, meta.xmax),
  ylim: (meta.ymin, meta.ymax),
  aspect-ratio: 1,
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
