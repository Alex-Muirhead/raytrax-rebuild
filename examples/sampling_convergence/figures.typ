// Figures of the sampling convergence study, drawn from sampling_convergence.py's data/.

#import "../style.typ": *

// RMS error against ray count, per cell and binned, with the seed range shaded.
#let rmse() = {
  let planar = lq.load-txt(read("data/slab_planar_rmse.csv"), header: true)
  let axisymmetric = lq.load-txt(read("data/slab_axisymmetric_rmse.csv"), header: true)
  let coarse = lq.load-txt(read("data/cylinder_coarse_rmse.csv"), header: true)
  lq.diagram(
    width: 9cm,
    height: 6cm,
    xscale: "log",
    yscale: "log",
    xlabel: [Rays],
    ylabel: [RMSE / mean heating],
    legend: (position: left + top, dx: 100% + 0.5em),

    lq.fill-between(planar.rays, planar.cell_lo, y2: planar.cell_hi, fill: cycle.at(0).transparentize(80%), stroke: none),
    lq.plot(planar.rays, planar.cell_rmse, mark: none, stroke: 1.2pt + cycle.at(0), label: [`slab_planar`, per cell]),
    lq.fill-between(planar.rays, planar.bin_lo, y2: planar.bin_hi, fill: cycle.at(0).transparentize(80%), stroke: none),
    lq.plot(
      planar.rays,
      planar.bin_rmse,
      mark: none,
      stroke: (paint: cycle.at(0), thickness: 1.2pt, dash: "dashed"),
      label: [`slab_planar`, binned],
    ),

    lq.fill-between(
      axisymmetric.rays,
      axisymmetric.cell_lo,
      y2: axisymmetric.cell_hi,
      fill: cycle.at(1).transparentize(80%),
      stroke: none,
    ),
    lq.plot(
      axisymmetric.rays,
      axisymmetric.cell_rmse,
      mark: none,
      stroke: 1.2pt + cycle.at(1),
      label: [`slab_axisymmetric`, per cell],
    ),
    lq.fill-between(
      axisymmetric.rays,
      axisymmetric.bin_lo,
      y2: axisymmetric.bin_hi,
      fill: cycle.at(1).transparentize(80%),
      stroke: none,
    ),
    lq.plot(
      axisymmetric.rays,
      axisymmetric.bin_rmse,
      mark: none,
      stroke: (paint: cycle.at(1), thickness: 1.2pt, dash: "dashed"),
      label: [`slab_axisymmetric`, binned],
    ),

    lq.fill-between(coarse.rays, coarse.cell_lo, y2: coarse.cell_hi, fill: cycle.at(2).transparentize(80%), stroke: none),
    lq.plot(
      coarse.rays,
      coarse.cell_rmse,
      mark: none,
      stroke: 1.2pt + cycle.at(2),
      label: [`cylinder_coarse`, per cell],
    ),
    lq.fill-between(coarse.rays, coarse.bin_lo, y2: coarse.bin_hi, fill: cycle.at(2).transparentize(80%), stroke: none),
    lq.plot(
      coarse.rays,
      coarse.bin_rmse,
      mark: none,
      stroke: (paint: cycle.at(2), thickness: 1.2pt, dash: "dashed"),
      label: [`cylinder_coarse`, binned],
    ),

    // One guide, on the planar slab. Where a curve leaves it, rays have
    // stopped being what limits the answer.
    lq.plot(planar.rays, planar.cell_sqrt_law, mark: none, stroke: 0.6pt + palette.grey, label: $N^(-1\/2)$),
  )
}

#standalone((rmse: rmse))
