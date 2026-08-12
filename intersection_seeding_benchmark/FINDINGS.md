# Findings: bilinear-seeded vs. naive Newton seeding on the pipe elbow

Run: 16 elbow boundary patches (8 flat end-caps, 8 curved walls), 12×12 parameter
samples × 4 incidence angles (0/40/65/82°) per patch = 9 216 rays, each strategy,
at 3 bilinear tolerances. Newton: max 50 iters, residual tol 1e-10. Fixed RNG.
Data: [`results.csv`](results.csv) (55 296 rows); reproduce with
[`analyze_results.py`](analyze_results.py).

## Headline

**Bilinear seeding is justified for this geometry — as a robustness measure first,
a speed win second.** In the high-curvature elbow bend it cuts the Newton failure
rate from **31% to ~6–9%** *and* lowers wall-clock time; in the flat end-caps it
halves iteration count with no failures either way.

## Aggregate (fail% = no valid hit at the true location)

| tol | region | strategy | mean it | max it | fail% | seed-miss% | Newton µs | seed µs | total µs |
|----:|:-------|:---------|--------:|-------:|------:|-----------:|----------:|--------:|---------:|
| 0.1 | flat | naive    | 5.34 | 6  | 0.0  | –    | 51.3 | –   | 51.3 |
| 0.1 | flat | bilinear | 2.66 | 3  | 0.0  | 0.0  | 25.7 | 2.6 | 28.4 |
| 0.1 | bend | naive    | 4.75 | 50 | 31.4 | –    | 45.4 | –   | 45.4 |
| 0.1 | bend | bilinear | 3.11 | 24 | 9.3  | 13.9 | 29.8 | 3.4 | 33.3 |
| 0.5 | bend | bilinear | 3.13 | 13 | 5.8  | 9.4  | 29.9 | 2.4 | 32.4 |
| 0.02| bend | bilinear | 3.29 | 50 | 22.8 | 39.8 | 32.8 | 4.5 | 37.4 |

Naive is tolerance-independent (identical across tol rows) — a good harness sanity
check. Bilinear total wall-clock (seed lookup + Newton) beats naive Newton-only in
every cell: the iteration savings exceed the 2–5 µs BVH lookup.

## The result that matters: failure vs. incidence angle (bend region)

| incidence | naive fail | bilinear fail @tol 0.1 | bilinear seed-miss @tol 0.1 |
|----------:|-----------:|-----------------------:|----------------------------:|
| 0°  | 21.0% | **1.2%** | 5.7% |
| 40° | 23.6% | **1.1%** | 5.5% |
| 65° | 34.8% | **2.8%** | 4.2% |
| 82° | 46.3% | 32.1%    | 40.4% |

For all but extreme-grazing rays, bilinear seeding on the curved walls collapses
Newton failure from 21–35% to 1–3%. The naive midpoint seed fails often even at
*normal* incidence here because the walls sweep 90° over a `v∈[0,2]` (multi-span)
domain, so the domain midpoint is a poor guess far from most hits.

## Counterintuitive: finer tolerance is *worse* for seeding

The plan hypothesized finer tolerance ⇒ better seed ⇒ fewer failures. **The opposite
holds.** Bend failure grows as the approximation is refined:

| tol | bend fail% | bend seed-miss% |
|----:|-----------:|----------------:|
| 0.5 (coarse) | **5.8** | 9.4 |
| 0.1 (medium) | 9.3 | 13.9 |
| 0.02 (fine)  | 22.8 | 39.8 |

Cause: at fine tolerance the leaves are thin, near-planar slivers whose exact AABBs
hug the surface. Rays that only *approach* the true surface (grazing, and even some
normal-incidence rays that pass within ~tol of a sliver) thread between the boxes and
the BVH returns **no leaf** — those rays fall back to the naive midpoint and inherit
its failures. The effect is uniform across all 8 wall patches (33–44% miss at fine
tol), and padding every AABB by 0.05 (2.5× the tolerance) does *not* recover them, so
the gaps are real, not a token-padding artifact. A seed oracle wants a **fat,
forgiving** approximation, not an accurate one — the opposite of what you want when
tracing directly against the bilinear surface.

**Operational sweet spot: the coarse/medium approximation (tol 0.1–0.5).** It gives
the cheapest lookup, the fewest seed misses, and the best robustness.

## Answering the underlying question

- **Iteration count:** bilinear seeding roughly halves mean iterations everywhere
  (flat 5.3→2.5–3.3; bend 4.75→3.1) and, more importantly, cuts the worst-case tail
  (bend max 50→13 at tol 0.5).
- **Robustness:** the decisive win — bend failures 31%→6% (tol 0.5), and ~1–3% for
  non-grazing rays at tol 0.1. This is the reason to adopt it.
- **Wall-clock:** favorable — bilinear total (lookup + reduced Newton) is lower than
  naive Newton-only in every cell.

### Verdict
Adopt bilinear-patch seeding for ray/NURBS intersection on this class of curved,
multi-span patches, using a **coarse** bilinear approximation. Its value is primarily
robustness in high-curvature regions; the modest speed-up is a bonus. Its one blind
spot is extreme-grazing rays (~82°), where the approximation is missed and the solve
falls back to naive — worth a dedicated fallback (e.g. a coarse UV grid search) for
that regime rather than a finer approximation, which makes coverage *worse*.

## Caveats / follow-ups
- The fine-tolerance coverage gap (BVH misses that a 0.05 AABB pad doesn't fix) may
  also indicate thin-sliver / watertightness behavior in the conforming reduction at
  small budgets — orthogonal to seeding, but worth a look for the direct bilinear
  ray tracer.
- Timings include per-iteration `std::vector` allocations in the evaluator; absolute
  µs are inflated but the naive-vs-bilinear *ratio* is unaffected (same evaluator).
- Rays are constructed to pass exactly through a known point at a set incidence, so
  every ray has ground truth; real render rays would add primary-visibility culling.
