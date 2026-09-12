"""
Statistical comparison of the two reward functions (default vs. freshness_bonus) on
late_gold_avg_delay, across every seed actually present on disk for each condition.

This exists because a 5-seed paired one-tailed t-test landed at p=0.047 - right at the
conventional 0.05 line, with only 4 degrees of freedom. That is fragile: one seed going the
other way could flip the conclusion. This script is the re-check after adding 5 more seeds
per condition (seeds 5-9), and it is deliberately built to not trust anything from memory:
every number it reports is re-read from the dqn_evaluation_seed*.csv files on disk, via the
same glob-based discovery multiseed_study.py itself uses for its aggregate stats - not from
any previously-computed summary CSV, and not from a running process's memory.

Hypothesis and direction. "Better" = LOWER late_gold_avg_delay (the metric the whole
multi-seed study exists to protect - see multiseed_study.py's docstring). The alternative
hypothesis tested is one-tailed: mean(default) > mean(freshness_bonus), i.e. the freshness
bonus REDUCES late-Gold delay. This matches the direction of the original 5-seed test
(p=0.047 in favour of the freshness bonus), so the same question is being asked, not a
different one dressed up as a re-check.

Two tests are reported, because they answer different questions:
  - PAIRED (matched by seed number): removes seed-to-seed variance common to both
    conditions (the same seed drives the same job-arrival randomness and epsilon-greedy
    exploration draws in both runs), so it is the more sensitive test and the one the
    original p=0.047 came from.
  - UNPAIRED / Welch's: treats the two conditions as independent samples, ignoring the
    seed-number pairing entirely. Less sensitive, but does not assume the pairing structure
    is doing anything - a useful cross-check that does not lean on that assumption.

Run:
    python3 src/analyze_reward_comparison.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy import stats

from environment import CSV_DIR
from multiseed_study import discover_seed_evaluations

DEFAULT_DIR = CSV_DIR / "multiseed"
FRESHNESS_DIR = CSV_DIR / "multiseed" / "freshness_bonus"

METRIC = "late_gold_avg_delay"
ALPHA_LEVELS = [0.05, 0.01]


def load_metric(out_dir: Path, metric: str = METRIC) -> pd.Series:
    """seed -> metric value, read directly from every dqn_evaluation_seed*.csv in out_dir."""
    rows = discover_seed_evaluations(out_dir)  # {seed: {column: value, ...}} from disk, not memory
    return pd.Series({seed: row[metric] for seed, row in rows.items()}, name=metric).sort_index()


def verdict(p: float) -> str:
    if p < 0.01:
        return "CONFIRMED (p < 0.01)"
    if p < 0.05:
        return "CONFIRMED (p < 0.05)"
    if p < 0.10:
        return "STILL SUGGESTIVE BUT NOT CONCLUSIVE (0.05 <= p < 0.10)"
    return "NOT SUPPORTED (p >= 0.10)"


def main() -> None:
    default = load_metric(DEFAULT_DIR)
    freshness = load_metric(FRESHNESS_DIR)

    print("=" * 100)
    print(f"REWARD COMPARISON: {METRIC}  (default vs. freshness_bonus)")
    print("=" * 100)
    print(f"  default reward      seeds on disk: {list(default.index)}  (n={len(default)})")
    print(f"  freshness_bonus     seeds on disk: {list(freshness.index)}  (n={len(freshness)})")

    paired_seeds = sorted(set(default.index) & set(freshness.index))
    only_default = sorted(set(default.index) - set(freshness.index))
    only_freshness = sorted(set(freshness.index) - set(default.index))
    if only_default:
        print(f"  seeds ONLY in default (no freshness_bonus counterpart, excluded from "
              f"the paired test): {only_default}")
    if only_freshness:
        print(f"  seeds ONLY in freshness_bonus (excluded from the paired test): {only_freshness}")

    print(f"\n  Matched pairs used for the paired test: {paired_seeds}  (n={len(paired_seeds)})")

    table = pd.DataFrame({"default": default, "freshness_bonus": freshness}).loc[paired_seeds]
    table["default_minus_freshness"] = table["default"] - table["freshness_bonus"]
    table["freshness_better"] = table["default_minus_freshness"] > 0
    with pd.option_context("display.float_format", lambda v: f"{v:,.2f}"):
        print("\n  Per-seed late_gold_avg_delay (both conditions, re-read from disk):")
        print(table.to_string())

    n_better = int(table["freshness_better"].sum())
    print(f"\n  freshness_bonus had LOWER (better) late-Gold delay in {n_better} of "
          f"{len(paired_seeds)} seeds.")

    # ---- PAIRED one-tailed t-test (matched by seed) --------------------------------------
    paired_t, paired_p = stats.ttest_rel(table["default"], table["freshness_bonus"],
                                         alternative="greater")
    paired_df = len(paired_seeds) - 1

    # ---- UNPAIRED (Welch's) one-tailed t-test, ALL seeds in each condition, not just the
    # ---- overlap - this is what "unpaired" means: it does not require a matched partner.
    welch_t, welch_p = stats.ttest_ind(default, freshness, equal_var=False, alternative="greater")
    # Welch-Satterthwaite degrees of freedom, for reporting alongside the p-value.
    v1, v2, n1, n2 = default.var(ddof=1), freshness.var(ddof=1), len(default), len(freshness)
    welch_df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))

    print("\n" + "=" * 100)
    print("HYPOTHESIS TEST: does the freshness bonus reduce late-Gold delay?")
    print("H1 (one-tailed): mean(default) > mean(freshness_bonus)")
    print("=" * 100)
    print(f"\n  PAIRED t-test (matched by seed, n={len(paired_seeds)}, df={paired_df}):")
    print(f"    t = {paired_t:.4f}, p = {paired_p:.4f}")
    print(f"    -> {verdict(paired_p)}")

    print(f"\n  UNPAIRED Welch's t-test (default n={n1}, freshness_bonus n={n2}, "
          f"df~={welch_df:.1f}):")
    print(f"    t = {welch_t:.4f}, p = {welch_p:.4f}")
    print(f"    -> {verdict(welch_p)}")

    # ---- explicit, unrounded comparison against the original 5-seed result --------------
    print("\n" + "=" * 100)
    print("HAS THE EXPANDED SAMPLE CONFIRMED, WEAKENED, OR REVERSED THE ORIGINAL FINDING?")
    print("=" * 100)
    print("  Original (seeds 0-4, paired, one-tailed): p = 0.047 (at the edge of significance)")
    print(f"  Now      (seeds {paired_seeds}, paired, one-tailed): p = {paired_p:.4f}")
    if paired_p < 0.047:
        direction = ("STRENGTHENED - the larger sample pushed the paired p-value below the "
                     "original figure.")
    elif paired_p < 0.05:
        direction = ("HELD, but only barely - still under 0.05, but not meaningfully "
                     "stronger than the fragile original result.")
    elif paired_p < 0.10:
        direction = ("WEAKENED - no longer under the conventional 0.05 threshold on the "
                     "paired test with the expanded sample.")
    else:
        direction = ("REVERSED FROM SIGNIFICANT TO NOT - the original edge-of-significance "
                     "result did not survive more seeds.")
    print(f"  -> {direction}")

    # ---- does seed 4 remain the hardest seed, or was that a small-sample artifact? --------
    print("\n" + "=" * 100)
    print("SEED 4: consistently the hardest seed in 0-4 under both conditions - does this "
          "hold for 5-9?")
    print("=" * 100)
    for name, series in (("default", default), ("freshness_bonus", freshness)):
        ranked = series.sort_values(ascending=False)
        rank_of_4 = list(ranked.index).index(4) + 1 if 4 in ranked.index else None
        hardest_seed = ranked.index[0]
        print(f"\n  {name}:")
        print(f"    full ranking (hardest first): "
              + ", ".join(f"seed{ s }={ranked[s]:,.1f}" for s in ranked.index))
        if rank_of_4 is not None:
            print(f"    seed 4 rank: {rank_of_4} of {len(ranked)}"
                 + ("  <- still the single hardest seed" if rank_of_4 == 1 else
                    "  <- no longer the hardest; a seed from 5-9 is now harder"
                    if hardest_seed != 4 else ""))
        else:
            print("    seed 4 not present in this condition's data.")

    seeds_5_9_present = [s for s in range(5, 10) if s in default.index or s in freshness.index]
    if not seeds_5_9_present:
        print("\n  NOTE: no seeds 5-9 were found on disk yet - this report reflects "
              "whatever partial data is currently available, not the full 10-seed study.")
    elif len(seeds_5_9_present) < 5:
        print(f"\n  NOTE: only seeds {seeds_5_9_present} of the intended 5-9 were found on "
              f"disk - the full run may still be in progress.")


if __name__ == "__main__":
    main()
