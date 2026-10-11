// Figures of the march benchmark, drawn from performance.py's data/.

#import "../style.typ": *

// Lane-steps per second against the lanes marched together.
#let throughput() = {
  let planar = lq.load-txt(read("data/slab_planar.csv"), header: true)
  let axisymmetric = lq.load-txt(read("data/slab_axisymmetric.csv"), header: true)
  let cylinder = lq.load-txt(read("data/cylinder.csv"), header: true)
  let capsule = lq.load-txt(read("data/reentry.csv"), header: true)
  let y-scale = y => y / 1e9
  lq.diagram(
    width: 9cm,
    height: 6cm,
    // Lane counts are powers of two.
    xscale: lq.scale.log(base: 2),
    xlabel: [Lanes],
    ylabel: [Throughput steps [#quan[GHz]]],
    legend: (position: left + top, dx: 100% + 0.5em),
    lq.xaxis(
      position: top,
      scale: "log",
    ),
    lq.plot(planar.lanes, planar.lane_steps_per_s.map(y-scale), stroke: 1.2pt + cycle.at(0), color: cycle.at(0), label: [`slab (planar)`]),
    lq.plot(
      axisymmetric.lanes,
      axisymmetric.lane_steps_per_s.map(y-scale),
      stroke: 1.2pt + cycle.at(1),
      color: cycle.at(1),
      label: [`slab (axisymmetric)`],
    ),
    lq.plot(cylinder.lanes, cylinder.lane_steps_per_s.map(y-scale), stroke: 1.2pt + cycle.at(2), color: cycle.at(2), label: [`cylinder`]),
    lq.plot(capsule.lanes, capsule.lane_steps_per_s.map(y-scale), stroke: 1.2pt + cycle.at(3), color: cycle.at(3), label: [`reentry`]),
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

// Time of a whole trace against the rays traced, on the GPU and the CPU.
// Solid is the median run, with bars from the fastest to the slowest; dashed
// adds the compile that the first run pays.
#let crossover() = {
  let gpu = lq.load-txt(read("data/backend_gpu.csv"), header: true)
  let cpu = lq.load-txt(read("data/backend_cpu.csv"), header: true)
  let rays = json("data/crossover.json").rays
  let total(data) = data.median_s.zip(data.compile_s).map(((run, compile)) => run + compile)
  let spread(data) = (
    p: data.max_s.zip(data.median_s).map(((slow, median)) => slow - median),
    m: data.median_s.zip(data.min_s).map(((median, fast)) => median - fast),
  )
  lq.diagram(
    width: 9cm,
    height: 6cm,
    xscale: "log",
    yscale: "log",
    xlabel: [Rays],
    ylabel: [Wall time [#quan[s]]],
    legend: (position: left + top, dx: 100% + 0.5em),
    ..if rays != none { (lq.vlines(rays, stroke: (paint: palette.grey, dash: "dotted")),) },
    lq.plot(
      gpu.rays,
      gpu.median_s,
      yerr: spread(gpu),
      stroke: 1.2pt + palette.blue,
      color: palette.blue,
      label: [GPU],
    ),
    lq.plot(
      cpu.rays,
      cpu.median_s,
      yerr: spread(cpu),
      stroke: 1.2pt + palette.orange,
      color: palette.orange,
      label: [CPU],
    ),
    lq.plot(
      gpu.rays,
      total(gpu),
      stroke: (paint: palette.blue, thickness: 1.2pt, dash: "dashed"),
      mark: none,
      label: [GPU, with compile],
    ),
    lq.plot(
      cpu.rays,
      total(cpu),
      stroke: (paint: palette.orange, thickness: 1.2pt, dash: "dashed"),
      mark: none,
      label: [CPU, with compile],
    ),
  )
}

#standalone((throughput: throughput, compile-time: compile-time, crossover: crossover))
