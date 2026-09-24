// Figures of the march benchmark, drawn from performance.py's data/.

#import "../style.typ": *

#let cases = ("slab_planar", "slab_axisymmetric", "cylinder", "reentry")

#let ladder(value, ylabel: none, yscale: "linear") = lq.diagram(
  width: 9cm,
  height: 6cm,
  // Lane counts are powers of two.
  xscale: lq.scale.log(base: 2),
  yscale: yscale,
  xlabel: [Lanes],
  ylabel: ylabel,
  legend: (position: left + top, dx: 100% + 0.5em),
  ..cases
    .enumerate()
    .map(((index, name)) => {
      let rows = records(csv("data/" + name + ".csv"))
      lq.plot(
        column(rows, "lanes"),
        column(rows, value),
        stroke: 1.2pt + cycle.at(index),
        color: cycle.at(index),
        label: raw(name),
      )
    }),
)

// Lane-steps per second against the lanes marched together.
#let throughput() = ladder("lane_steps_per_s", ylabel: [Throughput (lane-steps #quan[/s])], yscale: "log")

// Compile time against the lanes marched together.
#let compile-time() = ladder("compile_s", ylabel: labelled([Compile time], "s"))

#let figures = (throughput: throughput, compile-time: compile-time)

#standalone(figures)
