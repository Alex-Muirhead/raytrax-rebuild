#set heading(numbering: "1.")
#set par(justify: true)

= Axisymmetric path tracing

A 2D mesh can be read two ways. *Planar* extends the domain infinitely out of
plane, and the only correction needed is to the distance travelled. *Axisymmetric*
revolves the domain about an axis, and then a straight 3D ray no longer projects
to a straight line in the mesh: it projects to a #emph[hyperbola].

This note derives the crossing that the hyperbola requires, and shows that the
quadratic it produces factors through the ray's two asymptotes, which determine
the crossing completely: the trace's intersections with a face plane follow from
where the two asymptotes meet that plane, and from nothing else.

Nothing here is implemented yet. #ref(<seams>) lists the seams it needs.

== Geometry

Let the meridional half-plane have in-plane coordinates $(x, r)$, with $x$ along
the axis and $r >= 0$ the radius. #ref(<axis>) covers how those map onto the
coordinates actually stored in the mesh.

A ray is a straight line in 3D, $arrow(P)(t) = arrow(P)_0 + t arrow(d)$, with
$|arrow(d)| = 1$. Split off the axial component: write $arrow(P) = (x, arrow(p))$
with $arrow(p)$ the 2-vector perpendicular to the axis, and
$arrow(d) = (d_x, arrow(e))$, and set

$ u := |arrow(e)|^2 = 1 - d_x^2 $

Take the current point as $t = 0$; `advance` moves the ray to each crossing, so the
solve is always local to the cell. Write $x_0$, $arrow(p)_0$,
$r_0 = |arrow(p)_0|$ and $m := arrow(p)_0 dot arrow(e)$. Then

$ x(t) = x_0 + d_x t, quad quad r(t)^2 = u t^2 + 2 m t + r_0^2 $

=== It is a hyperbola

Completing the square, with $t_m = -m \/ u$ and
$b^2 = r_0^2 - m^2 \/ u = (|arrow(p)_0 times arrow(e)|^2) \/ u$,

$ r^2 = u (t - t_m)^2 + b^2 $

so in the meridional plane, with $x_m = x_0 + d_x t_m$ and
$a = b |d_x| \/ sqrt(u)$,

$ r^2/b^2 - (x - x_m)^2/a^2 = 1 $

Here $b$ is the ray's closest approach to the axis. The hyperbola is centred at
$(x_m, 0)$ and opens along $r$, with #emph[asymptote directions]

$ arrow(s)_+ = (d_x, +sqrt(u)), quad quad arrow(s)_- = (d_x, -sqrt(u)) $

which are unit vectors, since $d_x^2 + u = 1$.

=== Two structural facts <structure>

*The trace lies on or above both asymptotes.* From
$r^2 = u(t - t_m)^2 + b^2 >= u(t - t_m)^2$, with equality only when $b = 0$. The
asymptotes bound the path from below and are reached only in the degenerate
no-swirl case.

*The trace is convex.* $r(x)$ is the square root of a positive-definite quadratic,
hence convex. A convex curve meets a convex set in a single connected arc, so the
trace enters and leaves each cell #emph[exactly once]. The march never needs
re-entry bookkeeping.

The corollary is a trap. That single exit may lie on the #emph[same face as the
entry], when the hyperbola's vertex falls inside the cell and the arc bends back
out the way it came. This is routine in cells next to the axis, and any "reject
roots near $t = 0$" tolerance will silently eat it.

#figure(
  block(width: 10.4cm, height: 3.35cm, {
    let sx = 1.45cm
    let sy = 1.35cm
    let top = 2.3
    let span = 3.4
    let pt(x, r) = (sx * (x + span), sy * (top - r))
    let b = 0.8
    let k = 0.55
    let lid = 1.35
    let hyp(x) = calc.sqrt(b * b + k * k * x * x)
    let edge = calc.sqrt(lid * lid - b * b) / k
    let cell = 2.6

    // the cell, a band against the axis
    place(curve(
      fill: luma(94%), stroke: 0.5pt + luma(55%),
      curve.move(pt(-cell, 0)), curve.line(pt(cell, 0)),
      curve.line(pt(cell, lid)), curve.line(pt(-cell, lid)), curve.close(),
    ))
    // axis of revolution
    place(line(start: pt(-span, 0), end: pt(span, 0), stroke: 1pt + black))
    // asymptotes
    for s in (-1, 1) {
      place(line(
        start: pt(0, 0), end: pt(s * span, k * span),
        stroke: (paint: luma(40%), thickness: 0.5pt, dash: "dashed"),
      ))
    }
    // the trace
    let n = 120
    let pts = range(n + 1).map(i => {
      let x = -span + 2 * span * i / n
      pt(x, hyp(x))
    })
    place(curve(
      stroke: 1.4pt + rgb("#a02c2c"),
      curve.move(pts.at(0)),
      ..pts.slice(1).map(p => curve.line(p)),
    ))
    // entry and exit, both on the lid
    for s in (-1, 1) {
      place(dx: pt(s * edge, lid).at(0) - 2pt, dy: pt(s * edge, lid).at(1) - 2pt,
            circle(radius: 2pt, fill: rgb("#a02c2c"), stroke: none))
    }

    let label(x, r, body, ..a) = place(dx: pt(x, r).at(0), dy: pt(x, r).at(1), ..a, body)
    label(-edge, lid + 0.34, text(8pt)[entry], dx: pt(-edge, lid).at(0) - 0.62cm)
    label(edge, lid + 0.34, text(8pt)[exit], dx: pt(edge, lid).at(0) + 0.12cm)
    label(0, b - 0.42, text(8pt)[$b$], dx: pt(0, b).at(0) - 0.28cm)
    place(line(start: pt(0, 0), end: pt(0, b), stroke: (thickness: 0.5pt, dash: "dotted")))
    label(span - 0.05, k * span + 0.3, text(8pt)[$arrow(s)_+$])
    label(-span + 0.05, k * span + 0.3, text(8pt)[$arrow(s)_-$], dx: -0.1cm)
    label(-span + 0.05, -0.02, text(8pt)[$r = 0$], dx: -0.15cm)
  }),
  caption: [The trace turns inside a cell next to the axis, so it enters and leaves
    through the same face. Dashed: the asymptotes, which it never reaches.],
) <turning>

=== Degenerate configurations

/ $b = 0$: no swirl ($arrow(p)_0 parallel arrow(e)$). The hyperbola collapses onto
  its asymptotes; the ray stays in one meridional plane, touches the axis and
  reflects off it.
/ $d_x = 0$: ray perpendicular to the axis. $arrow(s)_(plus.minus) = (0, plus.minus 1)$
  and the trace is the vertical line $x = x_0$.
/ $u = 0$: ray along the axis. The two asymptotes coincide at
  $arrow(s)_(plus.minus) = (plus.minus 1, 0)$ and $r equiv r_0$ is constant.

== The crossing <crossing>

A face slot carries an outward-signed meridional unit normal
$arrow(n) = (n_x, n_r)$ and offset $c$, being `scene.normal` and
`scene.offset` exactly as they stand today. We want the smallest $t > 0$ with

$ g(t) := n_x x(t) + n_r r(t) - c = 0, quad quad (d g)/(d t) > 0 $

Isolating the radical gives $n_r r(t) = L(t)$, where $L(t) = L_0 - n_x d_x t$ and
$L_0 = c - n_x x_0$. Squaring yields a quadratic $A t^2 + 2 B t + C = 0$ with

$ A &= n_r^2 u - n_x^2 d_x^2 = n_r^2 - d_x^2 quad (arrow(n) "unit") \
  B &= n_r^2 m + n_x d_x L_0 \
  C &= n_r^2 r_0^2 - L_0^2 = g_0 (n_r r_0 + L_0) $

where $g_0 = n_x x_0 + n_r r_0 - c$ is the signed distance, non-positive inside.

=== The asymptote identity

$ A = -(arrow(n) dot arrow(s)_+)(arrow(n) dot arrow(s)_-) $

Write $sigma_(plus.minus) = arrow(n) dot arrow(s)_(plus.minus)$. Each is precisely
the `alignment` that `LinearPath.crossing` already computes, evaluated for a
straight ray along that asymptote.

The first thing this buys is the case analysis, read off two dot products with no
discriminant:

#table(
  columns: 3,
  table.header([*alignments*], [*$A$*], [*crossings of the plane*]),
  [$sigma_+ sigma_- > 0$ (asymptotes agree)], [$< 0$],
  [exactly one; a guaranteed exit if both are positive],
  [$sigma_+ sigma_- < 0$ (asymptotes straddle)], [$> 0$],
  [zero or two: the trace is trapped in the wedge the plane cuts],
  [$sigma_+ approx 0$ or $sigma_- approx 0$], [$approx 0$],
  [degenerate; a single linear root $t = -C\/(2B)$],
)

Two checks against intuition. For a face perpendicular to the axis,
$arrow(n) = (1, 0)$, both alignments equal $d_x$ and $A = -d_x^2$; there is one
crossing, and the quadratic collapses to $-(d_x t - L_0)^2 = 0$, which is the
linear answer. For a cylindrical face $r = R$, $arrow(n) = (0, 1)$, the alignments
are $plus.minus sqrt(u)$ and straddle; a line misses a cylinder or hits it twice.

The third row is the reason to keep the factored form. Testing
$|sigma_(plus.minus)|$ against a tolerance is well conditioned; testing $A$
directly is not.

=== The asymptotes determine the crossing <asymptote-solve>

The identity classifies, but the asymptotes do more than classify: they determine
the crossing outright, which is the full strength of the sketch in `README.typ`.

Work in asymptotic coordinates. Let $arrow(f)_0$ be the centre and
$arrow(f)_1, arrow(f)_2$ the two asymptote directions, both pointing away from the
axis. Then

$ arrow(p)(v) = arrow(f)_0 + v/2 arrow(f)_1 + 1/(2 v) arrow(f)_2, quad quad v > 0 $

traces the branch: the two coefficients multiply to a constant, which is the
hyperbola written in the frame of its own asymptotes. Substituting $v = e^theta$
recovers $arrow(f)_0 + (B sinh theta, A cosh theta)$ with $A = b$ and
$B = d_x b \/ sqrt(u)$, and $v > 0$ selects the branch at positive radius.

Now intersect each asymptote #emph[line] with the face plane, at the parameters

$ beta_i = (c - arrow(n) dot arrow(f)_0) / (arrow(n) dot arrow(f)_i) $

Call those points $A' = arrow(f)_0 + beta_1 arrow(f)_1$ and
$B' = arrow(f)_0 + beta_2 arrow(f)_2$. Every point of the chord $A' B'$ has the
form $arrow(f)_0 + (beta_1 (1 + sigma))/2 arrow(f)_1 + (beta_2 (1 - sigma))/2 arrow(f)_2$,
so matching it against $arrow(p)(v)$ requires both
$v = beta_1 (1 + sigma)$ and $v = [beta_2 (1 - sigma)]^(-1)$. Eliminating $v$
leaves $beta_1 beta_2 (1 - sigma^2) = 1$, and therefore

$ v = beta_1 plus.minus sqrt(beta_1^2 - beta_1 \/ beta_2) $

Both crossings, and whether they exist at all, follow from $beta_1$ and $beta_2$
alone. Nothing about the hyperbola beyond its two asymptotes enters. Intersecting
with the asymptotes #emph[is] enough.

=== The same quadratic in different clothing

Writing $g_c$ for the signed distance of the centre from the face plane, and with
$|arrow(f)|^2 = A^2 + B^2$,

$ beta_1 beta_2 = g_c^2 / (|arrow(f)|^2 A) $

so $"sign"(beta_1 beta_2) = "sign"(A)$ and the table reappears: $beta_1 beta_2 < 0$
is the single-crossing row, and $beta_1 beta_2 > 0$ crosses only if
$beta_1 beta_2 >= 1$, a condition carrying the same sign as $B^2 - A C$. The
discriminant is not extra information beyond the asymptotes; it #emph[is] the
asymptote data.

The geometry behind that threshold: $beta_1 beta_2$ is the ratio of the area of
triangle $O A' B'$ to that of $O arrow(f)_1 arrow(f)_2$, whose chord is tangent to
the vertex. Below 1 the chord passes through the throat and misses.

Two things the asymptote form does not remove. It still yields two roots, and the
outgoing one must still be chosen (#ref(<root-selection>)). And it solves against
the face's #emph[plane], not the bounded face: $beta_i$ is unrestricted, so $A'$
and $B'$ may fall outside the face's extent. In half-space form that is exactly
what is wanted, but it would matter if faces were ever clipped to their true
extent.

Which form to implement is a numerical question, not a mathematical one. The
$beta$ form divides by $arrow(n) dot arrow(f)_i$, which vanishes when an asymptote
is parallel to the face, where the quadratic instead degenerates gracefully to a
linear root. Prefer the quadratic in the kernel, and this derivation as the reason
it is correct.

=== Root selection <root-selection>

Squaring admits roots on the mirrored face ($r -> -r$), which is the hyperbola's
other branch. Two filters remove them, both pure sign tests with no magnitude
tolerance:

+ *Branch.* Keep $t$ with $n_r L(t) >= 0$, since $n_r r = L$ and $r >= 0$.
+ *Outgoing.* Keep $t$ with $n_x d_x r(t) + n_r (u t + m) > 0$, which is
  $r dot (d g)/(d t)$.

Take the smallest survivor, then reduce across slots as the linear path does.

The second filter is what makes the entry-face case of #ref(<structure>) work. On
the face a ray has just entered, $t = 0$ is a root but has $(d g)/(d t) <= 0$, so
it is rejected, while a genuine second crossing of that same plane survives.
This is why the test is on the sign of the derivative and not on the magnitude of
$t$.

This is the failure the old implementation hit: for steep trajectories in the row
of cells against the axis, the same boundary is met twice, and choosing between
the roots by their sign returns the crossing #emph[into] the cell where the one
#emph[out] of it was wanted. The ray then skips that cell and charges its path
length to the previous one. Selecting on $(d g)/(d t)$ makes the choice on the
quantity that actually distinguishes the two.

=== Numerics

Use the stable pairing rather than the textbook formula: with
$q = -(B + "sign"(B) sqrt(B^2 - A C))$, the roots are $q\/A$ and $C\/q$.
Branch to the linear root when $|sigma_(plus.minus)|$ falls below tolerance.

Form $C$ as $g_0 (n_r r_0 + L_0)$, not as $n_r^2 r_0^2 - L_0^2$. On a face the
first degrades to nearly zero; the second degrades to cancellation noise of order
$r_0^2$.

Cost is about fifteen flops and one square root per face slot. $sqrt(u)$ and $m$
are per-ray and computed once. The shape is identical to the linear path, so the
`vmap`, the padded null half-spaces and the `nanargmin` reduction are all
untouched.

== Fitting the existing code <seams>

The ray stays a straight 3D line throughout; the hyperbola appears only inside
`crossing`. Three consequences, all favourable:

- `advance` is `position += t * tangent`. Exact, with no integration drift.
- `travel` accumulates $t$, which #emph[is] the true 3D path length. So
  `transport.step`'s $exp(-kappa s)$ needs no correction factor at all.
  Axisymmetric is simpler here than planar, which does need one.
- Reflection off a wall is genuinely 3D and falls out of the existing formula.

=== The 2D-mesh, 3D-ray mismatch

An axisymmetric ray state is 3D while `scene.dim` is 2. Three sites outside
`paths.py` assume those are the same number: `sources.sample` draws its direction
from `sphere(key, scene.dim)`, `sources.sample` returns the sampled mesh point as
the position, and the `diffuse` branch of `interactions.apply` builds a
hemisphere about the meridional face normal.

Put the lifting on `Path`, as three members that are pure coordinate plumbing and
leave every distribution choice where it already lives:

/ `ndim`: dimension of the ray state: 3 for axisymmetric, `scene.dim` otherwise.
/ `lift(point)`: mesh point to ray-space position. Rotational symmetry lets a
  newly emitted ray always start at $theta = 0$, so $(x, r) -> (x, r, 0)$. Only
  new points need it; rays in flight already carry a 3D position.
/ `lift_normal(state, normal)`: meridional normal to ray-space normal. This one
  cannot be a constant transform, because the 3D outward normal of an
  axisymmetric wall depends on where the ray sits on the ring:
  $ arrow(n)_3 = (n_x, n_r hat(p)), quad quad hat(p) = arrow(p)/(|arrow(p)|) $
  `reflect` needs the identical expression, which is the argument for naming it
  once. `LinearPath` gets identities throughout.

The alternative, a symmetry flag on `Scene` that `sources` and `interactions` branch
on, needs fewer members but spreads axisymmetric knowledge across three
modules, and every future mode would touch all of them. That is the opposite of
how `interactions.apply` keeps boundary kinds out of `transport`.

=== Choosing the axis <axis>

An axis of revolution is a #emph[line], so $+x$ and $-x$ are the same axis and
there is nothing to distinguish. The genuinely free choices are:

+ Which in-plane coordinate is axial, the other being the radius. One static
  field on `AxisymmetricPath`.
+ The sign of the radial coordinate, for a mesh stored at negative radius. One
  more static field and a multiply.
+ Which plane the 2D mesh occupies. This one is not in `paths.py`:
  `mesh/geometry.py` slices `vertices[:, :topology.dim]`, so a 2D mesh is
  truncated to its first two columns and a mesh lying in $x z$ would be
  flattened. Expressing "revolve about $z$" at all means making that column
  selection configurable in the reader.

Default to axial = first in-plane coordinate, radius = second, positive.

=== Summary of touched files

#table(
  columns: 2,
  table.header([*file*], [*change*]),
  [`paths.py`], [add `AxisymmetricPath`; add `ndim`, `lift`, `lift_normal` to
    `Path`; identities on `LinearPath`],
  [`sources.py`], [`sample` gains a `path` argument, threading through
    `transport._refill` and `transport.trace`],
  [`interactions.py`], [the `diffuse` branch routes through `path.lift_normal`
    instead of building a `RayState` directly],
  [`mesh/geometry.py`, `io/vtk.py`], [configurable in-plane column selection, if
    meshes off the $x y$ plane are wanted],
  [`transport.py`], [none],
)

`RayState`'s arrays become 3-vectors while `scene.dim` is 2. That asymmetry is
deliberate and belongs in the module docstring.

== What still blocks a meaningful run <geometry-work>

Stated plainly, because it is easy to miss: with the crossing alone an
axisymmetric run still produces wrong numbers, because the geometry remains
Cartesian. Two things are needed before any energy check can pass.

*Pappus weighting.* A cell is a ring, and a face is a surface of revolution:
$ V = 2 pi macron(r)_c A_"poly", quad quad S = 2 pi macron(r)_f L_"seg" $
Exact for polygons, and `mesh/geometry.py` already computes the exact area and
centroid from its simplex decomposition, so this is a multiplication.

*Radius-weighted sampling.* A point uniform in the ring is #emph[not] uniform in
the meridional plane; the density carries a factor $r$. Exactly: choose the
sub-triangle with probability proportional to $macron(r) A$, then within a
triangle the density $prop sum_i lambda_i r_i$ is a mixture, so with probability
$r_i\/(sum_j r_j)$ draw $arrow(lambda) tilde "Dirichlet"(arrow(alpha))$ with
$alpha_i = 2$ and the rest 1.

== Verification

+ *Scanned reference.* For random rays and random #emph[bounded] convex cells,
  the returned $t$ matches a dense scan of $max_i g_i$.
+ *Degenerate coverage.* One case each for $b = 0$, $d_x = 0$, $u approx 0$ and
  $sigma_(plus.minus) approx 0$.
+ *Entry-face re-crossing.* A ray entering an axis-adjacent cell whose vertex
  falls inside must exit through the face it entered by, at the scanned $t$.
+ *Axis unreachable.* No ray terminates on the $r = 0$ group unless $b$ is below
  tolerance. Guards against leaks through the symmetry axis.
+ *End-to-end*, once #ref(<geometry-work>) is done. An axisymmetric cylinder must
  reproduce the existing 3D cylinder heating, binned in $(x, r)$, within
  Monte-Carlo noise. This is the strongest check available and it reuses
  `examples/tracing.py` directly.

A warning for whoever writes the first test. Random sets of half-spaces are
frequently #emph[unbounded], and a ray legitimately never exits one. While
drafting this note, 26 apparent failures in 500 were all of that kind. Constrain
the normals to positively span the plane.

=== Results so far

The identities and the selection rule above were checked numerically against a
dense scan of the true path before being written down:

/ Asymptote identity: $A = n_r^2 - d_x^2$ and $A = -sigma_+ sigma_-$ agreed
  exactly over 400 random configurations, with no mismatches.
/ Root selection: over 600 random bounded convex cells, the smallest surviving
  root matched a dense scan of $max_i g_i$ to $10^(-5)$, with no mismatches.
/ Entry-face re-crossing: 300 of 300 constructed cases recovered the scanned
  value.
/ Asymptote form (#ref(<asymptote-solve>)): over 20000 random ray and face-plane pairs,
  $v = beta_1 plus.minus sqrt(beta_1^2 - beta_1 \/ beta_2)$ reproduced the
  quadratic's root set exactly, agreeing on existence in every case.
/ Bridge identities: $beta_1 beta_2 = g_c^2 \/ (|arrow(f)|^2 A)$,
  $"sign"(beta_1 beta_2) = "sign"(A)$, and $"sign"(beta_1 beta_2 - 1) =
  "sign"(B^2 - A C)$ for $A > 0$ each held over 20000 configurations.
