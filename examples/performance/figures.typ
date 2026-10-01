// Figures of the march benchmark, drawn from performance.py's data/.

#import "../style.typ": *

// Lane-steps per second against the lanes marched together.
#let throughput() = {
  let planar = lq.load-txt(read("data/slab_planar.csv"), header: true)
  let axisymmetric = lq.load-txt(read("data/slab_axisymmetric.csv"), header: true)
  let cylinder = lq.load-txt(read("data/cylinder.csv"), header: true)
  let capsule = lq.load-txt(read("data/reentry.csv"), header: true)
  lq.diagram(
    width: 9cm,
    height: 6cm,
    // Lane counts are powers of two.
    xscale: lq.scale.log(base: 2),
    yscale: "log",
    xlabel: [Lanes],
    ylabel: [Throughput (lane-steps #quan[/s])],
    legend: (position: left + top, dx: 100% + 0.5em),
    lq.plot(planar.lanes, planar.lane_steps_per_s, stroke: 1.2pt + cycle.at(0), color: cycle.at(0), label: [`slab_planar`]),
    lq.plot(
      axisymmetric.lanes,
      axisymmetric.lane_steps_per_s,
      stroke: 1.2pt + cycle.at(1),
      color: cycle.at(1),
      label: [`slab_axisymmetric`],
    ),
    lq.plot(cylinder.lanes, cylinder.lane_steps_per_s, stroke: 1.2pt + cycle.at(2), color: cycle.at(2), label: [`cylinder`]),
    lq.plot(capsule.lanes, capsule.lane_steps_per_s, stroke: 1.2pt + cycle.at(3), color: cycle.at(3), label: [`reentry`]),
  )
}

// Compile time against the lanes marched together. Every distinct lane count is a fresh compile.
#let compile-time() = {
  let planar = lq.load-txt(read("data/slab_planar.csv"), header: true)
  let axisymmetric = lq.load-txt(read("data/slab_axisymmetric.csv"), header: true)
  let cylinder = lq.load-txt(read("data/cylinder.csv"), header: true)
  let capsule = lq.load-txt(read("data/reentry.csv"), header: true)
  lq.diagram(
    width: 9cm,
    height: 6cm,
    xscale: lq.scale.log(base: 2),
    xlabel: [Lanes],
    ylabel: [Compile time [#quan[s]]],
    legend: (position: left + top, dx: 100% + 0.5em),
    lq.plot(planar.lanes, planar.compile_s, stroke: 1.2pt + cycle.at(0), color: cycle.at(0), label: [`slab_planar`]),
    lq.plot(
      axisymmetric.lanes,
      axisymmetric.compile_s,
      stroke: 1.2pt + cycle.at(1),
      color: cycle.at(1),
      label: [`slab_axisymmetric`],
    ),
    lq.plot(cylinder.lanes, cylinder.compile_s, stroke: 1.2pt + cycle.at(2), color: cycle.at(2), label: [`cylinder`]),
    lq.plot(capsule.lanes, capsule.compile_s, stroke: 1.2pt + cycle.at(3), color: cycle.at(3), label: [`reentry`]),
  )
}

#standalone((throughput: throughput, compile-time: compile-time))
