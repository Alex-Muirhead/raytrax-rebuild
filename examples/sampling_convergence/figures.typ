// Figures of the sampling convergence study, drawn from sampling_convergence.py's data/.

#import "../style.typ": *

#let cases = ("slab_planar", "slab_axisymmetric", "cylinder_coarse")

// RMS error against ray count, per cell and binned, with the seed range shaded.
#let rmse() = {
  let plots = ()
  for (index, name) in cases.enumerate() {
    let rows = records(csv("data/" + name + "/rmse.csv"))
    let rays = column(rows, "rays")
    let colour = cycle.at(index)
    for (metric, dash, label) in (("cell", none, "per cell"), ("bin", "dashed", "binned")) {
      plots.push(lq.fill-between(
        rays,
        column(rows, metric + "_lo"),
        y2: column(rows, metric + "_hi"),
        fill: colour.transparentize(80%),
        stroke: none,
      ))
      plots.push(lq.plot(
        rays,
        column(rows, metric + "_rmse"),
        mark: none,
        stroke: (paint: colour, thickness: 1.2pt, dash: dash),
        label: raw(name) + [, #label],
      ))
    }
  }
  // One guide, on the first case. Where a curve leaves it, rays have stopped
  // being what limits the answer.
  let first = records(csv("data/" + cases.first() + "/rmse.csv"))
  plots.push(lq.plot(
    column(first, "rays"),
    column(first, "cell_sqrt_law"),
    mark: none,
    stroke: 0.6pt + palette.grey,
    label: $N^(-1\/2)$,
  ))
  lq.diagram(
    width: 9cm,
    height: 6cm,
    xscale: "log",
    yscale: "log",
    xlabel: [Rays],
    ylabel: [RMSE / mean heating],
    legend: (position: left + top, dx: 100% + 0.5em),
    ..plots,
  )
}

#let figures = (rmse: rmse)

#standalone(figures)
