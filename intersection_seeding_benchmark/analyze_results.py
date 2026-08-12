#!/usr/bin/env python3
"""Analyze the seeding-benchmark CSV: aggregate metrics + iteration histograms.

Reads results.csv (one row per ray x strategy x tolerance) and reports, split by
region (flat vs. bend) and overall, for each tolerance:

  * mean / median / worst-case (max) iteration count, naive vs. bilinear
  * failure rate (no valid on-patch hit at the correct location)
  * total wall-clock cost: naive (Newton only) vs. bilinear (seed lookup + Newton)
  * iteration-count histograms (text + optional PNG)

Then prints a verdict against the underlying question for each tolerance.

Usage:
    python3 intersection_seeding_benchmark/analyze_results.py \
        [intersection_seeding_benchmark/results.csv]
"""

import csv
import os
import sys
from collections import defaultdict

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


def load(path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            r["tol"] = float(r["tol"])
            r["iters"] = int(r["iters"])
            r["success"] = int(r["success"])
            r["valid_hit"] = int(r["valid_hit"])
            r["seed_hit"] = int(r["seed_hit"])
            r["incidence_deg"] = float(r["incidence_deg"])
            r["newton_ns"] = float(r["newton_ns"])
            r["seed_ns"] = float(r["seed_ns"])
            r["param_err"] = float(r["param_err"])
            rows.append(r)
    return rows


def pct(n, d):
    return 100.0 * n / d if d else 0.0


def median(xs):
    if not xs:
        return 0
    s = sorted(xs)
    return s[len(s) // 2]


def summarize(rows):
    tols = sorted({r["tol"] for r in rows})
    groups = ["flat", "bend", "overall"]

    print("=" * 100)
    print("AGGREGATE METRICS  (fail% = no valid hit at the true location)")
    print("=" * 100)
    header = (
        f"{'tol':>6} {'region':>7} {'strategy':>9} {'n':>6} "
        f"{'mean_it':>7} {'med_it':>6} {'max_it':>6} {'fail%':>6} "
        f"{'seedmiss%':>9} {'newton_us':>9} {'seed_us':>8} {'total_us':>9}"
    )
    for tol in tols:
        print(f"\n-- tolerance {tol} " + "-" * 80)
        print(header)
        for grp in groups:
            for strat in ["naive", "bilinear"]:
                sub = [
                    r
                    for r in rows
                    if r["tol"] == tol
                    and r["strategy"] == strat
                    and (grp == "overall" or r["group"] == grp)
                ]
                if not sub:
                    continue
                n = len(sub)
                its = [r["iters"] for r in sub]
                fails = sum(1 for r in sub if not r["success"])
                seedmiss = sum(1 for r in sub if r["strategy"] == "bilinear" and not r["seed_hit"])
                newton_us = sum(r["newton_ns"] for r in sub) / n / 1000.0
                seed_us = sum(r["seed_ns"] for r in sub) / n / 1000.0
                print(
                    f"{tol:>6} {grp:>7} {strat:>9} {n:>6} "
                    f"{sum(its)/n:>7.2f} {median(its):>6} {max(its):>6} "
                    f"{pct(fails, n):>6.1f} {pct(seedmiss, n):>9.1f} "
                    f"{newton_us:>9.2f} {seed_us:>8.2f} {newton_us+seed_us:>9.2f}"
                )


def histograms(rows):
    tols = sorted({r["tol"] for r in rows})
    print("\n" + "=" * 100)
    print("ITERATION-COUNT HISTOGRAMS (bend region; where seeding matters most)")
    print("=" * 100)
    for tol in tols:
        print(f"\n-- tolerance {tol}, bend region --")
        for strat in ["naive", "bilinear"]:
            its = [
                r["iters"]
                for r in rows
                if r["tol"] == tol and r["strategy"] == strat and r["group"] == "bend"
            ]
            if not its:
                continue
            counts = defaultdict(int)
            for x in its:
                counts[x] += 1
            n = len(its)
            print(f"  {strat}:")
            for k in sorted(counts):
                bar = "#" * max(1, int(60 * counts[k] / n))
                print(f"    {k:>3} it | {bar} {counts[k]} ({pct(counts[k], n):.1f}%)")


def plot_histograms(rows, outdir):
    if not HAVE_MPL:
        print("\n(matplotlib unavailable; skipping PNG histograms)")
        return
    tols = sorted({r["tol"] for r in rows})
    for grp in ["flat", "bend"]:
        fig, axes = plt.subplots(1, len(tols), figsize=(5 * len(tols), 4), squeeze=False)
        for ax, tol in zip(axes[0], tols):
            for strat, color in [("naive", "#d1495b"), ("bilinear", "#2e86ab")]:
                its = [
                    r["iters"]
                    for r in rows
                    if r["tol"] == tol
                    and r["strategy"] == strat
                    and r["group"] == grp
                ]
                if its:
                    ax.hist(
                        its,
                        bins=range(0, max(its) + 2),
                        alpha=0.6,
                        label=strat,
                        color=color,
                        edgecolor="white",
                    )
            ax.set_title(f"{grp}, tol={tol}")
            ax.set_xlabel("Newton iterations")
            ax.set_ylabel("rays")
            ax.legend()
        fig.tight_layout()
        path = os.path.join(outdir, f"hist_{grp}.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"wrote {path}")


def verdict(rows):
    tols = sorted({r["tol"] for r in rows})
    print("\n" + "=" * 100)
    print("VERDICT: is bilinear seeding justified?")
    print("=" * 100)
    for tol in tols:
        def stat(grp, strat):
            sub = [
                r
                for r in rows
                if r["tol"] == tol
                and r["strategy"] == strat
                and r["group"] == grp
            ]
            n = len(sub)
            fail = pct(sum(1 for r in sub if not r["success"]), n)
            mean_it = sum(r["iters"] for r in sub) / n if n else 0
            newton = sum(r["newton_ns"] for r in sub) / n / 1000.0 if n else 0
            seed = sum(r["seed_ns"] for r in sub) / n / 1000.0 if n else 0
            return fail, mean_it, newton, seed

        bf_n, bi_n, nn_n, ns_n = stat("bend", "naive")
        bf_b, bi_b, nn_b, ns_b = stat("bend", "bilinear")

        d_fail = bf_n - bf_b
        d_total = (nn_n) - (nn_b + ns_b)
        print(f"\ntol {tol}:")
        print(
            f"  bend failure   naive {bf_n:5.1f}%  ->  bilinear {bf_b:5.1f}%   "
            f"(Delta {d_fail:+.1f} pts)"
        )
        print(
            f"  bend mean iter naive {bi_n:5.2f}   ->  bilinear {bi_b:5.2f}"
        )
        print(
            f"  bend wallclock naive {nn_n:6.2f}us -> bilinear {nn_b+ns_b:6.2f}us "
            f"(seed {ns_b:.2f} + newton {nn_b:.2f}); Delta {d_total:+.2f}us"
        )
        notes = []
        if d_fail >= 3.0:
            notes.append("robustness WIN (fewer divergences in the bend)")
        else:
            notes.append("negligible robustness change")
        if d_total > 0:
            notes.append("also faster wall-clock")
        else:
            notes.append("slower wall-clock (lookup outweighs iter savings)")
        print("  => " + "; ".join(notes))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results.csv"
    )
    rows = load(path)
    print(f"loaded {len(rows)} rows from {path}")
    summarize(rows)
    histograms(rows)
    plot_histograms(rows, os.path.dirname(os.path.abspath(path)))
    verdict(rows)


if __name__ == "__main__":
    main()
