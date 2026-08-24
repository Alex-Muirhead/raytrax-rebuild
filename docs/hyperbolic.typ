#import "@preview/cetz:0.4.2"
#import "@preview/numty:0.0.5" as nt
#import "@preview/typpuccino:0.1.0": latte, frappe, macchiato, mocha
#import "@preview/physica:0.9.3" : *
#import "@preview/subpar:0.2.2"

#set text(lang: "en", region: "au")
#set heading(numbering: "1.")
#set par(justify: true)

#let arange(start, stop, step: 1) = {
  import calc : *
  let num = ceil((stop - start) / step)
  // Reassign stop
  let stop = start + (num - 1) * step
  nt.linspace(start, stop, num)
}

#let geomrange(start, stop, step: 1) = {
  import calc : *
  let num = ceil((stop - start) / step)
  // Reassign stop
  let stop = start + (num - 1) * step
  nt.geomspace(start, stop, num)
}

= The hyperbola in its asymptote frame

A parametrised function $va(p)$ can be defined along the hyperbola aligned along the $x$-axis
$
  va(p)(v) = va(f)_0 + v/2 va(f)_1 + 1/(2v) va(f)_2
  space.quad "where" space.quad
  va(f)_1 = vec(+b, a), space va(f)_2 = vec(-b, a)
$
The vectors $va(f)_1$ and $va(f)_2$ define the asymptotes of the hyperbola. Writing $v = e^theta$, the left and right halves of the branch are $theta in RR^-$ and $theta in RR^+$ respectively. Only $v > 0$ is considered, which is the branch at positive radius. The multiplier $v$ is referred to as the _exponential angle_, and $theta$ as the _hyperbolic angle_.

Throughout, $t$ denotes the ray's true 3D arclength and nothing else; #ref(<arclength>) relates it to $v$.

#subpar.grid(
  columns: 2,
cetz.canvas(length: 20mm, {
  import cetz.draw: *
  let point(fill: black, stroke: none, ..args) = {
    if stroke == none { stroke = fill }
    circle(radius: 0.03, ..args, fill: fill, stroke: stroke)
  }

  set-style(
    padding: 0.1,
  )

  // Define the hyperbola parameters

  let a = 0.5
  let b = 0.5
  let c = calc.sqrt(a*a + b*b)

  let f0 = (0, 0)
  let f1 = (+b, a)
  let f2 = (-b, a)

  // Draw the hyperbola foundations

  let H-right = nt.apply(nt.linspace(-1.9, 1.9, 11), t => {
    (b*calc.sinh(t), +a*calc.cosh(t))
  })
  let H-left = nt.apply(nt.linspace(-1.9, 1.9, 11), t => {
    (b*calc.sinh(t), -a*calc.cosh(t))
  })
  catmull(..H-right)
  catmull(..H-left, stroke: (dash: "densely-dashed", thickness: 0.7pt))

  for v in arange(1, 6, step: 1) {
    let p = nt.add(nt.mult(f1, v/2), nt.mult(f2, 1/(2*v)))
    let n = nt.normalize(nt.sub(nt.mult(f1, v/2), nt.mult(f2, 1/(2*v))))
    let tag = nt.add(p, nt.mult((-n.at(1), n.at(0)), 0.2))
    line(p, nt.add(p, nt.mult((-n.at(1), n.at(0)), 0.1)))
    content(tag, [$#v$])
  }

  for v-inv in arange(2, 6, step: 1) {
    let v = 1/v-inv
    let p = nt.add(nt.mult(f1, v/2), nt.mult(f2, 1/(2*v)))
    let n = nt.normalize(nt.sub(nt.mult(f1, v/2), nt.mult(f2, 1/(2*v))))
    let tag = nt.add(p, nt.mult((-n.at(1), n.at(0)), 0.2))
    content(tag, [$1/#v-inv$])
  }

  for t in arange(0.2, 6, step: 0.1) {
    let p = nt.add(nt.mult(f1, t/2), nt.mult(f2, 1/(2*t)))
    let n = nt.normalize(nt.sub(nt.mult(f1, t/2), nt.mult(f2, 1/(2*t))))
    line(p, nt.add(p, nt.mult((-n.at(1), n.at(0)), 0.05)))
  }

  content((0, +3*a), [$theta > 0$])
  content((0, -3*a), [$theta < 0$])

  group({
    set-style(stroke: (thickness: 0.7pt, paint: gray))
    let scale = 3.5
    line(nt.mult(f1, -scale), nt.mult(f1, +scale))
    line(nt.mult(f2, -scale), nt.mult(f2, +scale))
  })

  line(f0, nt.add(f0, f1), mark: (end: ">", fill: black), name: "f1")
  line(f0, nt.add(f0, f2), mark: (end: ">", fill: black), name: "f2")
  content("f1.end", anchor: "west", [$f_1$])
  content("f2.end", anchor: "east", [$f_2$])
}), 

cetz.canvas(length: 20mm, {
  import cetz.draw: *
  let point(fill: black, stroke: none, ..args) = {
    if stroke == none { stroke = fill }
    circle(radius: 0.03, ..args, fill: fill, stroke: stroke)
  }

  set-style(
    padding: 0.1,
  )

  // Define the hyperbola parameters

  let a = 0.5
  let b = 0.5
  let c = calc.sqrt(a*a + b*b)

  let f0 = (0, 0)
  let f1 = (+b, a)
  let f2 = (-b, a)

  // Draw the hyperbola foundations

  let H-right = nt.apply(nt.linspace(-1.9, 1.9, 11), u => {
    (b*calc.sinh(u), +a*calc.cosh(u))
  })
  let H-left = nt.apply(nt.linspace(-1.9, 1.9, 11), u => {
    (b*calc.sinh(u), -a*calc.cosh(u))
  })
  catmull(..H-right)
  catmull(..H-left, stroke: (dash: "densely-dashed", thickness: 0.7pt))

  for a in (
      sym.minus + "1", 
      sym.minus + "1.5", 
      sym.minus + "0.5", 
      "+0.5", 
      "+1", 
      "+1.5"
  ) {
    let t = calc.exp(float(a))
    let p = nt.add(nt.mult(f1, t/2), nt.mult(f2, 1/(2*t)))
    let n = nt.normalize(nt.sub(nt.mult(f1, t/2), nt.mult(f2, 1/(2*t))))
    let tag = nt.add(p, nt.mult((-n.at(1), n.at(0)), 0.1))
    line(p, nt.add(p, nt.mult((-n.at(1), n.at(0)), 0.1)))
    content(tag, [$#a$], anchor: "south")
  }

  for a in arange(-1.8, 1.9, step: 0.1) {
    let t = calc.exp(a)
    let p = nt.add(nt.mult(f1, t/2), nt.mult(f2, 1/(2*t)))
    let n = nt.normalize(nt.sub(nt.mult(f1, t/2), nt.mult(f2, 1/(2*t))))
    line(p, nt.add(p, nt.mult((-n.at(1), n.at(0)), 0.05)))
  }

  floating(content((0, +3.2*a), [$theta in RR$]))

  group({
    set-style(stroke: (thickness: 0.7pt, paint: gray))
    let scale = 3.5
    line(nt.mult(f1, -scale), nt.mult(f1, +scale))
    line(nt.mult(f2, -scale), nt.mult(f2, +scale))
  })

  line(f0, nt.add(f0, f1), mark: (end: ">", fill: black), name: "f1")
  line(f0, nt.add(f0, f2), mark: (end: ">", fill: black), name: "f2")
  content("f1.end", anchor: "west", [$f_1$])
  content("f2.end", anchor: "east", [$f_2$])
})
)

With $v = exp(theta)$
$
  va(p)(theta) &= va(f)_0 + (e^theta)/2 va(f)_1 + (e^(-theta))/2 va(f)_2 \
    &= va(f)_0 + (e^theta + e^(-theta))/2 vec(0, a) + (e^theta - e^(-theta))/2 vec(b, 0) \
    &= va(f)_0 + vec(b sinh(theta), a cosh(theta))
$

#pagebreak()

Define
$
  va(m) &= 1/2 (beta_1 va(f)_1 + beta_2 va(f)_2) \
  va(h) &= 1/2 (beta_1 va(f)_1 - beta_2 va(f)_2)
$

#subpar.grid(
  columns: 2,
  cetz.canvas(length: 20mm, {
    import cetz.draw: *
    let point(fill: black, stroke: none, ..args) = {
      if stroke == none { stroke = fill }
      circle(radius: 0.03, ..args, fill: fill, stroke: stroke)
    }
  
    set-style(
      padding: 0.1,
    )
  
    // Define the hyperbola parameters
  
    let a = 0.5
    let b = 0.5
    let c = calc.sqrt(a*a + b*b)
  
    let f0 = (0, 0)
    let f1 = (+b, a)
    let f2 = (-b, a)
  
    // Draw the hyperbola foundations
    let hyperbola = u => {
      (b*calc.sinh(u), +a*calc.cosh(u))
    }

    let exp-hyperbola = v => {
      (b/2*(v - 1/v), a/2*(v + 1/v))
    }

    let v-coord = 3.163
    point(exp-hyperbola(v-coord), radius: 0.1, fill: green)

    let u-coord = calc.ln(v-coord)
    let s-prime-coord = (v-coord * v-coord - 1) / (2 * v-coord)
    let s-coord = s-prime-coord * c

    point(hyperbola(u-coord), radius: 0.1, fill: red)
  
    let H-true = nt.apply(nt.linspace(-1.75, 1.75, 11), hyperbola)
    let H-false = nt.apply(nt.linspace(-1.75, 1.75, 11), t => {
      (b*calc.sinh(t), -a*calc.cosh(t))
    })
    catmull(..H-true, name: "hyperbola")
    catmull(..H-false, stroke: (dash: "dashed"))
  
    group({
      set-style(stroke: (thickness: 0.7pt, paint: gray))
      let scale = 3.0
      line(nt.mult(f1, -scale), nt.mult(f1, +scale))
      line(nt.mult(f2, -scale), nt.mult(f2, +scale))
    })
  
    line(f0, nt.add(f0, f1), mark: (end: ">", fill: black), name: "f1")
    line(f0, nt.add(f0, f2), mark: (end: ">", fill: black), name: "f2")
    content("f1.end", anchor: "west", [$va(f)_1$])
    content("f2.end", anchor: "east", [$va(f)_2$])
  
    // Define the vertices of the line-segment
  
    let v1 = (1.2, 0.8)
    let v2 = (0.5, 0.9)
    let edge = nt.sub(v2, v1)
  
    point(v1, stroke: latte.blue, fill: white, name: "v1")
    point(v2, stroke: latte.blue, fill: white, name: "v2")
  
    let b1 = nt.det((v1, edge)) / nt.det((f1, edge))
    let b2 = nt.det((v1, edge)) / nt.det((f2, edge))
  
    let mid = nt.add(f0, nt.add(
      nt.mult(f1, b1/2), nt.mult(f2, b2/2)
    ))
  
    let tangent = nt.add(
      nt.mult(f1, b1/2), nt.mult(f2, -b2/2)
    )
  
    point(mid, name: "M")
    content("M", anchor: "north-west", [$M$])
    line(f0, mid, mark: (end: ">"), stroke: 0.7pt, name: "m")
    content("m.end", anchor: "south", [$va(m)$])
  
    line(f0, tangent, mark: (end: ">"), stroke: 0.7pt, name: "h")
    content("h.end", anchor: "west", [$va(h)$])
  
    line(
      (0,0),
      nt.mult(f1, b1*calc.sqrt(a*a)),
      nt.mult(f2, b2*calc.sqrt(a*a)), 
      fill: latte.blue.transparentize(70%), stroke: none,
      name: "minArea"
    )
    line(
      "minArea.centroid", (rel: (30deg, -0.4)),
      stroke: (paint: latte.blue, thickness: 1.5pt),
      mark: (start: ">", fill: latte.blue),
      name: "minArea-tag"
    )
    content("minArea-tag.end", anchor: "east", [Area $= a b$])
  
    point(f0, name: "O")
    point(nt.mult(f1, b1), name: "A'")
    point(nt.mult(f2, b2), name: "B'")
    content("O", anchor: "north", [$O$])
    content("A'", anchor: "west", [$A'$])
    content("B'", anchor: "east", [$B'$])
    line("A'", "B'", stroke: latte.blue + 0.7pt, name: "AB")
    
    intersections("crossings", "AB", "hyperbola")
    for-each-anchor("crossings", (name) => {
      point("crossings." + name, fill: white, stroke: black)
    })
  }),
  cetz.canvas(length: 20mm, {
    import cetz.draw: *
    let point(fill: black, stroke: none, ..args) = {
      if stroke == none { stroke = fill }
      circle(radius: 0.03, ..args, fill: fill, stroke: stroke)
    }
  
    set-style(
      padding: 0.1,
    )
  
    // Define the hyperbola parameters
  
    let a = 0.5
    let b = 0.5
    let c = calc.sqrt(a*a + b*b)
  
    let f0 = (0, 0)
    let f1 = (+b, a)
    let f2 = (-b, a)
  
    // Draw the hyperbola foundations
  
    let H-true = nt.apply(nt.linspace(-1.75, 1.75, 11), t => {
      (b*calc.sinh(t), +a*calc.cosh(t))
    })
    let H-false = nt.apply(nt.linspace(-1.75, 1.75, 11), t => {
      (b*calc.sinh(t), -a*calc.cosh(t))
    })
    catmull(..H-true, name: "hyperbola")
    catmull(..H-false, stroke: (dash: "dashed"))
  
    group({
      set-style(stroke: (thickness: 0.7pt, paint: gray))
      let scale = 3.0
      line(nt.mult(f1, -scale), nt.mult(f1, +scale))
      line(nt.mult(f2, -scale), nt.mult(f2, +scale))
    })
  
    line(f0, nt.add(f0, f1), mark: (end: ">", fill: black), name: "f1")
    line(f0, nt.add(f0, f2), mark: (end: ">", fill: black), name: "f2")
    content("f1.end", anchor: "west", [$va(f)_1$])
    content("f2.end", anchor: "east", [$va(f)_2$])
  
    // Define the vertices of the line-segment
  
    let v1 = (0.8, 1.2)
    let v2 = (0.9, 0.5)
    let edge = nt.sub(v2, v1)
  
    point(v1, stroke: latte.blue, fill: white, name: "v1")
    point(v2, stroke: latte.blue, fill: white, name: "v2")
  
    let b1 = nt.det((v1, edge)) / nt.det((f1, edge))
    let b2 = nt.det((v1, edge)) / nt.det((f2, edge))
    let bias = calc.sqrt(1 - 1 / (b1*b2))
  
    let mid = nt.add(f0, nt.add(
      nt.mult(f1, b1/2), nt.mult(f2, b2/2)
    ))
  
    let tangent = nt.add(
      nt.mult(f1, b1/2), nt.mult(f2, -b2/2)
    )
  
    point(mid, name: "M")
    content("M", anchor: "south-east", [$M$])
    line(f0, mid, mark: (end: ">"), stroke: 0.7pt, name: "m")
    content("m.end", anchor: "west", [$va(m)$])
  
    line(f0, tangent, mark: (end: ">"), stroke: 0.7pt, name: "h")
    content("h.end", anchor: "south", [$va(h)$])
  
    point(f0, name: "O")
    point(nt.mult(f1, b1), name: "A'")
    point(nt.mult(f2, b2), name: "B'")
    content("O", anchor: "north", [$O$])
    content("A'", anchor: "west", [$A'$])
    content("B'", anchor: "west", [$B'$])
    line(v1, "B'", stroke: latte.blue + 0.7pt, name: "AB")
    
    intersections("crossings", "AB", "hyperbola")
    for-each-anchor("crossings", (name) => {
      point("crossings." + name, fill: white, stroke: black)
    })
  })
)

$
  A' &= va(f)_0 + va(m) + va(h)  
    & space.quad "and" space.quad 
    && B' &= va(f)_0 + va(m) - va(h) 
  \
  A &= va(f)_0 + va(m) + sigma va(h)
    & space.quad "and" space.quad 
    && B &= va(f)_0 + va(m) - sigma va(h)
$
Intersections to the hyperbola only happen along asymptotes with positive $beta$ values. Due to the axisymmetric nature of the grid, all cells are located $r >= 0$, and as such at least one $beta$ value will be positive.


To comply with the equation of a hyperbola
$
  va(f)(v) = va(f)_0 + v/2 va(f)_1 + 1/(2v) va(f)_2
  eq.triple va(f)_0 + (beta_1(1 + sigma))/2 va(f)_1 + (beta_2(1 - sigma))/2 va(f)_2
$

Find $sigma$ such that $v = beta_1(1 plus.minus sigma) = [beta_2 (1 minus.plus sigma)]^(-1)$ 
$
  1 &= beta_1 beta_2 (1 - sigma^2) \
  sigma &= plus.minus sqrt(1 - 1/(beta_1 beta_2))
$
Geometrically, the quantity $beta_1 beta_2$ represents the ratio of the area $triangle"OAB"$ to the area of ...
If this ratio is less than 1, then no solution is possible as the line $overline("AB")$ passes through the throat of the hyperbola. 

The points $A'$ and $B'$ can be found by an intersection of the lines $va(f)_1$ and $va(f)_2$ respectively with the edge $va(v)_(12)$. Specifically
$
  A' = va(f)_0 + beta_1 va(f)_1 = va(v)_1 + mu va(v)_(12)
$
Note that unlike the previous intersection algorithm, $mu in RR$ is not limited to $[0, 1]$.
$
  (va(f)_0 - va(v)_1 + beta_1 va(f)_1) and va(v)_(12) = 0 \
  beta_1 = ((va(v)_1 - va(f)_0) and va(v)_(12)) / (va(f)_1 and va(v)_(12)), space.quad
  beta_2 = ((va(v)_1 - va(f)_0) and va(v)_(12)) / (va(f)_2 and va(v)_(12))
$


== Choosing the outgoing root <root-selection>

When solving for the crossing, there are two solutions. In the bottom row of cells, for steep trajectories, it can intercept the same boundary twice. The code currently accepts the first solution of positive vs negative, which is the "right" most solution. 
For a trajectory travelling left to right, this means the solution when crossing _into_ the cell can default to the intersection that should be _out of_ the cell. 

This will cause it to _skip_ the cell at the bottom, and use the total length for the previous cell.

$
  ... &= (
    (va(v)_1 - va(f)_0) and va(v)_(12) plus.minus sqrt(((va(v)_1 - va(f)_0) and va(v)_(12))^2 - (va(f)_1 and va(v)_(12))(va(f)_2 and va(v)_(12)))
  ) / (va(f)_1 and va(v)_(12)) \
  ... &= beta_1 plus.minus (sqrt(((va(v)_1 - va(f)_0) and va(v)_(12))^2 - (va(f)_1 and va(v)_(12))(va(f)_2 and va(v)_(12)))
  ) / (va(f)_1 and va(v)_(12)) \
  beta_1 (1 plus.minus sigma) &= beta_1 plus.minus sqrt(beta_1^2 - beta_1/beta_2) \
$

The two roots multiply to $beta_1 \/ beta_2$ and sum to $2 beta_1$, so the stable
pairing applies directly: take the root of larger magnitude from the sign-safe
formula and the other from the product.

= From wedges to dot products

The mesh stores each face as an outward unit normal and a plane constant, not as
a pair of vertices, so the crossing is wanted in that form. The two are the same
statement. With $va(n) = R(va(v)_(12)) \/ |va(v)_(12)|$ and
$R(va(w)) = (w_y, -w_x)$, dividing numerator and denominator by $|va(v)_(12)|$
turns the wedge quotient into
$
  beta_i = (c - va(n) dot va(f)_0) / (va(n) dot va(f)_i)
$
`mesh/geometry.py` already builds a 2D face normal as exactly that quarter turn,
so no sign correction is needed anywhere.

Better still, $beta$ need not be formed at all. Substituting $va(p)(v)$ into
$va(n) dot va(p) = c$ and clearing the $1 \/ v$ gives a quadratic in $v$ directly,
$
  w_1 v^2 + 2 g_c v + w_2 = 0,
  space.quad w_i = va(n) dot va(f)_i,
  space.quad g_c = va(n) dot va(f)_0 - c
$
This matters because $va(n) dot va(f)_i$ vanishes when a face is parallel to an
asymptote, where $beta_i$ diverges but the equation above merely becomes linear.
Since $beta_1 beta_2 = g_c^2 \/ (w_1 w_2)$, the condition "no solution below 1"
is exactly a negative discriminant, $g_c^2 < w_1 w_2$.

= Anchoring the parametrisation at the ray <anchor>

The frame above is pinned to the hyperbola's centre $va(f)_0$, which for a
slender hyperbola sits arbitrarily far from the ray. The current position is then
recovered as a difference of large quantities and loses precision accordingly.

The construction has two free scalings that do not move the curve: $v -> v \/ mu$
with $va(f)_1 -> mu va(f)_1$ and $va(f)_2 -> va(f)_2 \/ mu$, since the two
coefficients still multiply to a constant. Choosing $mu = v_0$ places the ray's
own position at $v = 1$. Write $va(G)_i$ for the rescaled vectors and
$va(e)_1 = (d_x, sqrt(u))$, $va(e)_2 = (-d_x, sqrt(u))$ for the unit asymptote
directions, with $u = 1 - d_x^2$. Then
$
  va(G)_1 = g_1 va(e)_1, space.quad va(G)_2 = g_2 va(e)_2,
  space.quad g_1 = r (sqrt(u) + dot(r)) / u, space.quad g_2 = r (sqrt(u) - dot(r)) / u
$
and, since $va(p)(v) - va(p)_0 = ((v - 1)\/2) [va(G)_1 - va(G)_2 \/ v]$, the
crossing becomes a quadratic in the *step* $delta = v - 1$,
$
  W_1 delta^2 + (2 g_0 + W_1 - W_2) delta + 2 g_0 = 0,
  space.quad W_i = va(n) dot va(G)_i,
  space.quad g_0 = va(n) dot va(p)_0 - c
$
Here $g_0$ is the signed distance from the ray's own position to the face, which
is small and well conditioned, and $va(f)_0$ never appears.

== The swirl, and why it is carried <swirl>

The two coefficients satisfy $g_1 g_2 = (L \/ u)^2$, where
$L = r sqrt(u - dot(r)^2)$ is the swirl. Forming $u - dot(r)^2$ cancels for a ray
that nearly meets the axis, so the small coefficient is instead recovered from
the product: compute whichever of $g_1$, $g_2$ is a sum of like signs, and divide
into $(L\/u)^2$ for the other.

That works because $L$ never has to be recomputed. At launch it is read straight
off the 3D direction as $L = r |d_z|$, exactly. Thereafter it is conserved:
$L^2 = r^2 (1 - |va(tau)|^2)$, a wall leaves $r$ untouched and mirroring
preserves $|va(tau)|$, so a specular bounce cannot move it.

== Arclength <arclength>

The axial coordinate advances at a constant rate, $x = x_m + g_1 d_x sinh theta$
and also $x = x_0 + d_x t$, so the 3D arclength follows from $v$ with no
correction factor,
$
  Delta t = 1/2 [(g_1 v - g_2 / v) - (g_1 - g_2)]
$
The march therefore needs no separate distance calculation, and $Delta t$ is
monotone in $v$ because $dv(t, v) = (g_1 + g_2 \/ v^2) \/ 2 > 0$.

= Selecting the root <outgoing>

#ref(<root-selection>) is the trap, and in this parametrisation the rule is a
single sign test. Since $g(v) = g_0 + ((v-1)\/2)(W_1 - W_2 \/ v)$,
$
  dv(g, v) = 1/2 (W_1 - W_2 / v^2)
$
so keep the roots with $delta > 0$ and $v^2 W_1 > W_2$, and take the smallest.
The second test is what distinguishes the two crossings of a boundary met twice:
on the face the ray has just entered, $v = 1$ is a root but has
$dv(g, v) < 0$, while a genuine second crossing of the same plane survives.
Selecting on the derivative rather than on the ordering is the whole fix.

Padded face slots fall out for free: a zero normal makes $W_1 = W_2 = g_0 = 0$,
so the roots are not finite and the outgoing test reads $0 > 0$ besides.

= Fitting the mesh

== Reduced ray state

Nothing stores the ray as a 3D line. `RayState` carries the meridional position
$(x, r)$ and a *non-unit* tangent $va(tau) = (d_x, dot(r))$, whose deficit from
unit length is the swirl. A 3D ray has five degrees of freedom, revolution
removes one, and $(x, r, d_x, dot(r))$ is exactly the four that remain; only the
sign of $L$ is lost, and the meridional plane is a mirror symmetry, so it is
unobservable. `AxisymmetricPath` adds $L$ itself, for the reason in
#ref(<swirl>).

This is the convention the planar mode already uses, where the deficit stands for
$d_z$ instead. Consequently `travel` is the true 3D path length in both modes and
`transport.step` needs no correction factor, and `reflect` is shared: a wall's 3D
normal is $(n_x, n_r hat(p))$ with no azimuthal component, so a bounce mirrors the
meridional pair and leaves the swirl alone.

== Pappus weighting

A cell is a ring and a face a surface of revolution, so the Cartesian measures
are wrong until reweighted:
$
  V = 2 pi macron(r)_c A_"poly", space.quad S = 2 pi macron(r)_f L_"seg"
$
Exact for polygons, and `mesh/geometry.py` already computes each simplex centroid,
so this is a multiplication. `geometry.revolve` applies it to volumes, face areas
and the decomposition weights together.

== Radius-weighted sampling

A point uniform in the ring is *not* uniform in the meridional plane; the density
carries a factor $r$. Weighting the simplex weights by $macron(r)$ handles the
choice of sub-simplex, leaving the density within one, which is a mixture: choose
corner $i$ with probability $r_i \/ sum_j r_j$, then draw
$va(lambda) tilde "Dirichlet"(va(alpha))$ with $alpha_i = 2$ and the rest 1.

= What the code calls these <code-map>

`AxisymmetricPath` follows this document, but its identifiers spell the symbols
out in words instead of carrying the letters across. Both sides of the map are
provisional: $va(G)_i$ and $W_i$ are this document's own invention and are
standard in neither hyperbolic geometry nor axisymmetric transport, and the
names read no better. What follows is the correspondence as it stands.

== `_frame`

Returns the two asymptote directions and their coefficients, for
$va(p)(v) = va(p)_0 + (v-1)/2 (va(G)_1 - va(G)_2 \/ v)$ with
$va(G)_i = g_i va(e)_i$ (#ref(<anchor>)).

#table(
  columns: 3,
  table.header([*Name*], [*Symbol*], [*Meaning*]),
  [`forward`], [$va(e)_1 = (d_x, sqrt(u))$], [unit asymptote direction the ray recedes along as $v -> oo$],
  [`backward`], [$va(e)_2 = (-d_x, sqrt(u))$], [the other unit asymptote, approached as $v -> 0$],
  [`first`], [$g_1$], [coefficient on $va(e)_1$],
  [`second`], [$g_2$], [coefficient on $va(e)_2$],
  [`reduced`], [$u = 1 - d_x^2$], [squared speed off the axial direction],
  [`rate`], [$sqrt(u)$], [the asymptotes' radial rate $dot(r)$],
  [`scale`], [$g_1 g_2 = (L\/u)^2$], [the coefficient product the swirl fixes],
  [`near`, `far`], [--], [conditioning, not geometry],
)

`near` is whichever of $g_1$, $g_2$ is a sum of like signs, and `far` is the
other recovered from their product (#ref(<swirl>)). The branch on the sign of
$dot(r)$ puts the pair back in order, so `first` is always $g_1$.

== `crossing`

Solves $W_1 delta^2 + l delta + 2 g_0 = 0$ for the step $delta = v - 1$, and
returns the smallest outgoing root. Here $l = 2 g_0 + W_1 - W_2$; the letters
$a$, $b$, $c$ are already spoken for above.

#table(
  columns: 3,
  table.header([*Name*], [*Symbol*], [*Meaning*]),
  [`along`], [$W_1 = va(n) dot va(G)_1$], [face normal against the outgoing asymptote],
  [`against`], [$W_2 = va(n) dot va(G)_2$], [face normal against the incoming asymptote],
  [`distance`], [$g_0 = va(n) dot va(p)_0 - c$], [signed distance from the ray's own position to the face plane],
  [`linear`], [$l = 2 g_0 + W_1 - W_2$], [the coefficient of $delta$],
  [`discriminant`], [$l^2 - 8 g_0 W_1$], [negative where the face misses the branch],
  [`pair`], [$q = -(l + "sgn"(l) sqrt(l^2 - 8 g_0 W_1))\/2$], [the sign-safe numerator of the stable quadratic formula, not a geometric quantity],
  [`step`], [$delta$], [both roots, as $q\/W_1$ and $2 g_0\/q$],
  [`outgoing`], [$v^2 W_1 > W_2$], [the test $dv(g, v) > 0$ of #ref(<outgoing>)],
  [`best`], [$delta$], [smallest surviving root per face],
)

The returned parameter is $delta$, a step in exponential angle. It is a distance
only in the straight-ray mode, where the same interface returns one.

== `advance`

#table(
  columns: 3,
  table.header([*Name*], [*Symbol*], [*Meaning*]),
  [`angle`], [$v = e^theta$], [exponential angle at the new point],
  [`moved`], [$va(p)(v) - va(p)_0$], [meridional displacement],
  [`flown`], [$Delta t$], [the 3D arclength of #ref(<arclength>)],
  [`rate`], [$sqrt(u)$], [as in `_frame`, read back off `forward`],
  [`radial`], [$dot(r) = sqrt(u) (g_1 - g_2\/v^2)\/(g_1 + g_2\/v^2)$], [the new radial rate, $dv(r, v)$ over $dv(t, v)$],
)

= Results

Measured against a longdouble reference, bisecting $max_i g_i (t)$ on the
projected trace, over random rays in random bounded convex cells. Errors are in
the exit *position*, which is what the next cell consumes.

The controlling parameter is the closest approach $a$ to the axis relative to the
launch radius. For an isotropically sampled direction that ratio is $|sin phi.alt|$
with $phi.alt$ uniform, so $P(a\/r < x) = (2\/pi) arcsin(x)$, confirmed to three
digits over $2 times 10^7$ samples.

#table(
  columns: 5,
  table.header([*$a\/r$*], [*$P$(below)*], [*median*], [*max*], [*failures*]),
  [isotropic], [--], [$2.2 times 10^(-16)$], [$2.0 times 10^(-15)$], [0 / 500],
  [$10^(-2)$], [$6.4 times 10^(-3)$], [$1.4 times 10^(-16)$], [$4.8 times 10^(-15)$], [0 / 500],
  [$10^(-4)$], [$6.4 times 10^(-5)$], [$1.6 times 10^(-16)$], [$6.6 times 10^(-15)$], [0 / 500],
  [$10^(-8)$], [$6.4 times 10^(-9)$], [$1.6 times 10^(-16)$], [$1.3 times 10^(-14)$], [0 / 500],
  [$10^(-12)$], [$6.4 times 10^(-13)$], [$1.1 times 10^(-16)$], [$2.1 times 10^(-14)$], [0 / 500],
  [$10^(-16)$], [$6.4 times 10^(-17)$], [$1.1 times 10^(-16)$], [$3.7 times 10^(-15)$], [0 / 500],
)

Regenerate with `uv run python scripts/hyperbolic_accuracy.py`.

Three findings worth recording, because each cost an implementation attempt:

/ Anchoring matters more than the algebra: a centre-anchored frame
  (#ref(<anchor>)) loses the crossing entirely below $a\/r approx 10^(-8)$, at a
  rate of about one case in five, returning a silently wrong position rather than
  no answer. Re-anchoring at the ray fixes the near-axial regime outright.
/ Carrying the swirl: this removes what remains (#ref(<swirl>)). Recovering it
  from $1 - |va(tau)|^2$ reintroduces the cliff, because $a^2 \/ r^2$ then falls
  below machine epsilon. With it carried there is no cliff at any approach tested.
/ The $beta$ quotient is innocent: its division was suspected first and measured
  harmless. The stable pairing handles a face parallel to an asymptote with no
  special case at all.

For scale, the same comparison against the arclength-parametrised quadratic
$A t^2 + 2 B t + C = 0$ gives medians of $2.2 times 10^(-16)$ and maxima of
$10^(-13)$ to $10^(-12)$: the same answer, an order of magnitude looser.

= Verification

+ *Scanned reference.* For random rays and random _bounded_ convex cells, the
  returned crossing matches a dense scan of $max_i g_i$. Random half-space sets
  are frequently _unbounded_, and a ray then legitimately never exits; constrain
  the normals to positively span the plane.
+ *Entry-face re-crossing.* A ray entering an axis-adjacent cell whose vertex
  falls inside must exit through the face it entered by (#ref(<outgoing>)).
+ *Swirl conserved* across an interior crossing and across a specular bounce.
+ *Axis unreachable.* No energy is deposited on the $r = 0$ group.
+ *End-to-end.* A revolved meridional rectangle reproduces the 3D cylinder
  heating within Monte-Carlo noise. Measured at $kappa = 1$ with $4 times 10^5$
  rays: median $0.30%$ against the tangent slab, maximum $0.76%$, energy closing
  to $3 times 10^(-16)$ and exactly zero deposited on the axis.
