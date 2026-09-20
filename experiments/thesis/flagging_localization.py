
# =========================================================================
# IMPORTS
# =========================================================================

import json
import time

import matplotlib.pyplot as plt
import numpy as np

from runtime.experiment import Outputs, parser
from runtime.plotting import use_style
from runtime.report import write_report
from simulation.batching import chunks
from simulation.continuous import HestonParams
from simulation.evt import GapRule, build_procedures, sorted_cache, validate_names
from simulation.scenarios import Scenario, simulate_studentized, size_adjusted_calibration

use_style()

FAMILY_COLOUR = {"top spacing": "#2a78d6", "every spacing": "#eb6834", "reference": "#8a8a85"}
FAMILY_MARKER = {"top spacing": "o", "every spacing": "s", "reference": "D"}
FAMILY = {
    "exp": "top spacing", "renyi_single": "top spacing",
    "renyi_peel": "every spacing", "renyi": "every spacing", "renyi_deep": "every spacing",
    "gumbel": "reference",
}

LINES = [
    ("exp", "--"), ("renyi_single", "-."),
    ("renyi", "-"),
    ("gumbel", ":"),
]

CANDIDATES = {
    "exp":          GapRule("upper",  2,    "single"),
    "renyi_single": GapRule("signed", 2,    "single"),
    "renyi_peel":   GapRule("signed", None, "single"),
    "renyi":        GapRule("signed", None, "batch"),
    "renyi_deep":   GapRule("signed", None, "deepest"),
}

DISPLAY = {
    "exp":          "Exponential, upper tail",
    "renyi_single": "Top spacing, both tails",
    "renyi_peel":   "Rényi, one increment at a time",
    "renyi":        "Rényi, argmax cut",
    "renyi_deep":   "Rényi, deepest cut",
    "gumbel":       "Gumbel threshold",
}

validate_names(FAMILY, CANDIDATES, complete=True)
validate_names(DISPLAY, CANDIDATES, complete=True)

# =========================================================================
# SIMULATION
# =========================================================================

def score(flagged, truth, n_passes):
    tp = len(flagged & truth)
    return {
        "n_true": len(truth),
        "n_flagged": len(flagged),
        "tp": tp,
        "fp": len(flagged - truth),
        "fn": len(truth - flagged),
        "exact": float(flagged == truth),
        "any_flagged": float(len(flagged) > 0),
        "passes": n_passes,
    }


def summarize(trials):
    n = len(trials)
    tp = np.array([t["tp"] for t in trials], dtype=float)
    fp = np.array([t["fp"] for t in trials], dtype=float)
    n_true = np.array([t["n_true"] for t in trials], dtype=float)
    n_flagged = np.array([t["n_flagged"] for t in trials], dtype=float)

    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.where(n_true > 0, tp / np.maximum(n_true, 1), np.nan)
        fdr = np.where(n_flagged > 0, fp / np.maximum(n_flagged, 1), 0.0)

    return {
        "n_paths": n,
        "detection_rate": float(np.mean([t["any_flagged"] for t in trials])),
        "exact_recovery": float(np.mean([t["exact"] for t in trials])),
        "recall": float(np.nanmean(recall)) if np.any(n_true > 0) else float("nan"),
        "fdr": float(np.mean(fdr)),
        "any_false_positive": float(np.mean(fp > 0)),
        "mean_false_positives": float(fp.mean()),
        "max_false_positives": int(fp.max()) if n else 0,
        "p99_flagged": float(np.quantile(n_flagged, 0.99)),
        "max_flagged": int(n_flagged.max()) if n else 0,
        "mean_flagged": float(n_flagged.mean()),
        "mean_passes": float(np.mean([t["passes"] for t in trials])),
    }


def evaluate(n, heston, scenario, procedures, n_paths, rng, chunk=500, k_n_exponent=0.5):
    trials = {name: [] for name in procedures}
    for take in chunks(n_paths, chunk):
        z_hat, patterns = simulate_studentized(n, take, heston, scenario, rng, k_n_exponent)
        for i in range(take):
            truth = set(np.unique(patterns[i].grid_indices).tolist())
            column = z_hat[:, i]
            cache = sorted_cache(column)
            for name, procedure in procedures.items():
                flagged, passes = procedure(column, cache)
                trials[name].append(score(set(flagged.tolist()), truth, passes))
    return {name: summarize(rows) for name, rows in trials.items()}


def grid_scenarios():
    return [
        Scenario("null", 0, description="no jumps"),
        Scenario("single", 1, 2.0, 2.0, description="one jump at 2x the critical scale"),
        Scenario("pair_same", 2, 2.5, 2.5, "positive", description="two equal jumps, same tail"),
        Scenario("pair_opposite", 2, 2.5, 2.5, "alternating", description="two equal jumps, opposite tails"),
        Scenario("cluster5_tight", 5, 2.5, 2.5, "random", description="five equal jumps, random signs"),
        Scenario("cluster5_spread", 5, 1.5, 6.0, "random", description="five jumps, magnitudes 1.5x-6x"),
        Scenario("many10", 10, 2.0, 3.0, "random", description="ten jumps, magnitudes 2x-3x"),
    ]


def sweep_scenarios(n_jumps, multipliers):
    return [
        Scenario(f"N{n_jumps}_m{m:g}", n_jumps, m, m, "random",
                 description=f"{n_jumps} jumps at {m}x")
        for m in multipliers
    ]


def calibrate(n, heston, level, n_null, rng):
    rules = dict(CANDIDATES)
    criticals, gumbel_level = size_adjusted_calibration(
        n, heston, rules, level, n_null, rng,
    )
    procedures, used = build_procedures(
        CANDIDATES, n, level, criticals, gumbel_level=gumbel_level,
    )
    return procedures, used


# =========================================================================
# PLOTTING
# =========================================================================

def label_without_collisions(ax, points, fontsize=7, pad=1.0):
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    to_display = ax.transData.transform
    placed = []
    for x, y, name in sorted(points, key=lambda p: -p[1]):
        px, py = to_display((x, y))
        text = ax.text(0, 0, name, fontsize=fontsize, color="#3B3B3B", va="center")
        box = text.get_window_extent(renderer=renderer)
        w, h = box.width, box.height
        tx, ty = px + 7.0, py
        for (ox, oy, ow, oh) in placed:
            overlaps_x = tx < ox + ow + pad and ox < tx + w + pad
            overlaps_y = ty - h / 2 < oy + oh / 2 + pad and oy - oh / 2 < ty + h / 2 + pad
            if overlaps_x and overlaps_y:
                ty = oy - oh / 2 - h / 2 - 2 * pad
        placed.append((tx, ty, w, h))
        text.set_position(ax.transData.inverted().transform((tx, ty)))
        if abs(ty - py) > 1.0:
            ax.annotate("", xy=(x, y), xytext=ax.transData.inverted().transform((tx - 2.0, ty)),
                        arrowprops=dict(arrowstyle="-", linewidth=0.5, color="#B0B0AC"),
                        zorder=1)


def make_figure(sweep_rows, sweep_n):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, n_jumps in ((axes[0], 1), (axes[1], 5)):
        for name, style in LINES:
            rows = sorted(
                (r for r in sweep_rows if r["procedure"] == name and r["n_jumps"] == n_jumps),
                key=lambda r: r["multiplier"],
            )
            if not rows:
                continue
            fam = FAMILY[name]
            ax.plot([r["multiplier"] for r in rows], [r["recall"] for r in rows],
                    marker=FAMILY_MARKER[fam], markersize=4, linewidth=2.0,
                    color=FAMILY_COLOUR[fam], linestyle=style, label=DISPLAY[name])
        ax.set_xlabel("jump size / critical scale")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, linewidth=0.4, alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_title(f"{n_jumps} jump{'s' if n_jumps > 1 else ''}", fontsize=11)
    axes[0].set_ylabel("recall (share of true jumps flagged)")
    axes[0].legend(loc="lower right")

    ax = axes[2]
    n_jumps_pivot, multiplier_pivot = 5, 1.5
    seen = set()
    points = []
    for name, fam in FAMILY.items():
        rows = [r for r in sweep_rows if r["procedure"] == name
                and r["n_jumps"] == n_jumps_pivot and r["multiplier"] == multiplier_pivot]
        if not rows:
            continue
        row = rows[0]
        ax.scatter(row["fdr"], row["recall"], s=48, color=FAMILY_COLOUR[fam],
                   marker=FAMILY_MARKER[fam], zorder=3, edgecolor="white", linewidth=1.0,
                   label=fam if fam not in seen else None)
        seen.add(fam)
        points.append((row["fdr"], row["recall"], DISPLAY[name]))
    ax.set_xlabel("false discovery rate")
    ax.set_ylabel("recall")
    ax.grid(True, linewidth=0.4, alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title(f"5 jumps at {multiplier_pivot:g}x the critical scale, every procedure",
                 fontsize=11)
    ax.legend(loc="lower right")
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi + 0.95 * (hi - lo))
    label_without_collisions(ax, points)

    fig.suptitle(f"Detecting and flagging jumps (n = {sweep_n}, size-adjusted to 5%)")
    fig.tight_layout()
    return fig


# =========================================================================
# MAIN
# =========================================================================

def main():
    
    # =====================================================================
    # PARSING
    # =====================================================================

    p = parser(__doc__, "figures/thesis/flagging_localization",
               n_values=[1000, 4000, 16000], n_paths=2000, level=0.05, seed=0)
    p.add_argument("--n-null", type=int, default=8000)
    p.add_argument("--sweep-n", type=int, default=4000)
    p.add_argument("--sweep-multipliers", type=float, nargs="+",
                   default=[0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0])
    args = p.parse_args()
    out = Outputs(args, "localization")

    # =====================================================================
    # SIMULATION
    # =====================================================================

    if out.replotting:
        sweep_rows = json.loads(out.json.read_text(encoding="utf-8"))["sweep"]
        make_figure(sweep_rows, args.sweep_n).savefig(out.figure)
        print(f"Replotted {out.figure} from saved data")
        return

    heston = HestonParams()
    rng = np.random.default_rng(args.seed)
    grid_rows = []
    sweep_rows = []
    calibrations = []

    for n in args.n_values:
        print(f"== n = {n} ==", flush=True)
        t0 = time.time()
        procedures, used = calibrate(n, heston, args.level, args.n_null, rng)
        calibrations.append({"n": n, **{f"crit_{k}": v for k, v in used.items()}})
        print(f"   size-adjusted calibration on {args.n_null} null paths "
              f"in {time.time()-t0:.1f}s", flush=True)

        for scenario in grid_scenarios():
            t0 = time.time()
            summary = evaluate(n, heston, scenario, procedures, args.n_paths, rng)
            for name, row in summary.items():
                grid_rows.append({"n": n, "scenario": scenario.name, "procedure": name, **row})
            print(f"   -- {scenario.name} ({time.time()-t0:.1f}s)", flush=True)
            for name in procedures:
                row = summary[name]
                print(f"      {name:16} fire={row['detection_rate']:.3f}"
                      f" recall={row['recall']:.3f} fdr={row['fdr']:.3f}"
                      f" exact={row['exact_recovery']:.3f}"
                      f" meanFP={row['mean_false_positives']:.3f}"
                      f" maxFlag={row['max_flagged']:>3}", flush=True)

    print(f"\n== magnitude sweep at n = {args.sweep_n} ==", flush=True)
    procedures, used = calibrate(args.sweep_n, heston, args.level, args.n_null, rng)
    calibrations.append({"n": args.sweep_n, "role": "sweep",
                         **{f"crit_{k}": v for k, v in used.items()}})
    for n_jumps in (1, 5):
        for scenario in sweep_scenarios(n_jumps, args.sweep_multipliers):
            summary = evaluate(args.sweep_n, heston, scenario, procedures, args.n_paths, rng)
            multiplier = float(scenario.mag_low)
            for name, row in summary.items():
                sweep_rows.append({
                    "n": args.sweep_n, "n_jumps": n_jumps, "multiplier": multiplier,
                    "procedure": name, **row,
                })
            best = max(summary, key=lambda k: summary[k]["recall"])
            print(f"   N={n_jumps} m={multiplier:>4}  best recall: {best}"
                  f" ({summary[best]['recall']:.3f})", flush=True)

    # =====================================================================
    # OUTPUT
    # =====================================================================

    make_figure(sweep_rows, args.sweep_n).savefig(out.figure)
    write_report(
        out.report,
        title="Flagging rules: which increments get blamed",
        doc=__doc__,
        script=__file__,
        params={
            "n_values": args.n_values,
            "n_paths": args.n_paths,
            "n_null": args.n_null,
            "sweep_n": args.sweep_n,
            "sweep_multipliers": args.sweep_multipliers,
            "level": args.level,
            "seed": args.seed,
            "heston": heston,
            "candidates": {name: rule.name for name, rule in CANDIDATES.items()},
            "scenarios": {s.name: s.description for s in grid_scenarios()},
        },
        data={"grid": grid_rows,
              "sweep": sweep_rows,
              "calibration": calibrations},
    )
    print(f"\nSaved {out.json}, {out.figure} and {out.report}")


if __name__ == "__main__":
    main()
