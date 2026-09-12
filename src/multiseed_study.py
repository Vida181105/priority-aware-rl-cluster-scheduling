"""
Multi-seed variance study for the DQN cluster scheduler.

Step 1 of the documented Future Scope plan: late-Gold delay has been shown to swing
51 s -> 2,294 s across checkpoints of a SINGLE training run (see
results/csv/dqn_greedy_checkpoints_run2.csv), purely from stochasticity - epsilon-greedy
exploration, replay sampling, and the arrival order of otherwise-identical jobs at t=0.
Before any future reward or hyperparameter change can be judged to have helped, the natural
run-to-run variance has to be measured first, or a change that lands inside that noise band
would be reported as a finding when it is not one.

This script trains N independent DQN agents (different seeds, identical hyperparameters),
evaluates each against the four baselines, and reports mean/std/min/max for the metrics
that matter: gold_avg_scheduling_delay, late_gold_avg_delay, bronze_avg_waiting_time.

FILE COLLISION HANDLING. train() writes results/csv/dqn_greedy_checkpoints.csv to a
hardcoded path on every call, and every seed's call targets that same path. This script
copies that file to a seed-suffixed name immediately after each train() call returns and
before the next seed's train() call starts, so no seed's checkpoint trail is silently
overwritten. Every other output (training log, evaluation table, model weights) is written
directly to a seed-suffixed path by this script and never touches an unsuffixed name.

STEP 2 - COMPARING REWARD VARIANTS. --reward selects which reward function train() and
evaluate() use (default: the shipped default_reward). A variant other than "default" writes
every output into its own subdirectory, results/csv/multiseed/<reward>/ and
models/<reward>/, so it can NEVER collide with or overwrite Step 1's default-reward files,
which remain exactly where they are at the top level of results/csv/multiseed/.

STEP 3 - ACCUMULATING AGGREGATE STATS ACROSS SEPARATE INVOCATIONS. The aggregate table
(multiseed_aggregate_stats.csv) and the per-seed summary (multiseed_per_seed_summary.csv)
are DERIVED files, rebuilt on every run from every dqn_evaluation_seed*.csv actually present
in out_dir - not from the in-memory results of just the seeds passed via --seeds this time.
This is what makes `python3 src/multiseed_study.py --seeds 5 6 7 8 9` correctly EXTEND a
prior 5-seed aggregate to 10 seeds instead of silently replacing it with only the 5 just
trained (the per-seed files like dqn_evaluation_seed0.csv were always safe, since those use
seed-specific filenames and are never touched by a later seed's run - only the aggregate was
at risk of quietly discarding earlier seeds).

Trust (sanity-check-passed) status per seed is read the same way, from a small sidecar file
(dqn_sanity_seed<N>.csv) written alongside each seed's evaluation from now on. Seeds trained
before this sidecar existed have no such file; for those, trust is recovered from the
all_sanity_checks_passed column of a pre-existing multiseed_per_seed_summary.csv in the same
directory (which already carried that flag for every seed run under the old code) rather
than assumed.

Run:
    python3 src/multiseed_study.py                                        # 5 seeds x 50 episodes, default reward
    python3 src/multiseed_study.py --seeds 0 1 --episodes 3               # smoke test
    python3 src/multiseed_study.py --reward freshness_bonus               # Step 2: same 5 seeds, freshness bonus
    python3 src/multiseed_study.py --reward freshness_bonus --seeds 0 1 --episodes 3  # Step 2 smoke test
    python3 src/multiseed_study.py --seeds 5 6 7 8 9                      # Step 3: extends seeds 0-4 to 0-9
"""

from __future__ import annotations

import argparse
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dqn_agent import EVAL_EVERY, DQNAgent, evaluate, sanity_checks, train
from environment import CSV_DIR, MODEL_DIR, default_reward, ensure_output_dirs, freshness_bonus_reward

MULTISEED_CSV_DIR = CSV_DIR / "multiseed"

_SEED_NUM_RE = re.compile(r"seed(\d+)\.csv$")


def _display_path(p: Path) -> str:
    """Path relative to the project root for display, or the absolute path if p falls
    outside it (e.g. --out-dir/--weights-dir pointed at a scratch directory for testing)."""
    try:
        return str(p.relative_to(CSV_DIR.parent.parent))
    except ValueError:
        return str(p)

# Name -> reward function. "default" maps to default_reward explicitly (rather than None)
# so the printed reward name in train()'s banner is always informative, never "None".
REWARD_VARIANTS = {
    "default": default_reward,
    "freshness_bonus": freshness_bonus_reward,
}

# Metrics the aggregate table reports mean/std/min/max for.
HEADLINE_METRICS = ["gold_avg_scheduling_delay", "late_gold_avg_delay",
                    "bronze_avg_waiting_time"]


def run_one_seed(seed: int, episodes: int, eval_every: int, reward_name: str,
                 reward_fn, out_dir: Path, weights_dir: Path) -> dict:
    """
    Train and evaluate one seed, writing every output to a seed-suffixed path under
    out_dir/weights_dir (which are reward-variant-specific - see main()).

    Returns a dict with the seed, the DQN's evaluation row (as a dict), the sanity-check
    results, and whether all sanity checks passed - the caller decides whether to trust the
    seed's numbers based on that flag, per the user's requirement.
    """
    print("\n" + "=" * 100)
    print(f"SEED {seed}  reward={reward_name}  ({episodes} episodes, checkpoints every {eval_every})")
    print("=" * 100)
    t0 = time.time()

    agent, log = train(n_episodes=episodes, seed=seed, eval_every=eval_every, reward_fn=reward_fn)

    # ---- move the just-written checkpoint file before the NEXT seed's train() call can
    # ---- overwrite it. This happens synchronously, before run_one_seed returns, so there
    # ---- is no window in which two seeds' train() calls are both in flight. train() always
    # ---- writes to the SAME hardcoded top-level path regardless of reward variant, so this
    # ---- move is what actually keeps a freshness_bonus run from clobbering Step 1's
    # ---- default-reward checkpoint file (or a later seed clobbering an earlier one).
    unsuffixed = CSV_DIR / "dqn_greedy_checkpoints.csv"
    seed_checkpoints = out_dir / f"dqn_greedy_checkpoints_seed{seed}.csv"
    if unsuffixed.exists():
        shutil.move(str(unsuffixed), str(seed_checkpoints))
        print(f"  moved {unsuffixed.name} -> {_display_path(seed_checkpoints)}")
    else:
        # eval_every could in principle be set to skip every checkpoint (0 or > episodes);
        # train() then never writes the file. Not an error, just nothing to move.
        print(f"  NOTE: {unsuffixed.name} was not created this run (no checkpoints fired)")

    # ---- training log, seed-suffixed ----
    log_path = out_dir / f"dqn_training_log_seed{seed}.csv"
    log.to_csv(log_path, index=False)

    # ---- evaluation against the four baselines + this seed's DQN, seed-suffixed ----
    eval_df = evaluate(agent, seed=seed, reward_fn=reward_fn)
    eval_path = out_dir / f"dqn_evaluation_seed{seed}.csv"
    eval_df.to_csv(eval_path)

    # ---- sanity checks, run and required to pass before the result is trusted ----
    checks = sanity_checks(agent)
    all_passed = all(ok for _, ok, _ in checks)
    print(f"\n  Sanity checks (seed {seed}):")
    for name, ok, detail in checks:
        print(f"    [{'PASS' if ok else 'FAIL'}] {name:45s} {detail}")
    if not all_passed:
        print(f"  *** WARNING: seed {seed} FAILED one or more sanity checks - "
              f"its results should NOT be trusted without investigation ***")

    # Persist the trust verdict as its own seed-keyed sidecar (not just in this process's
    # memory), so a LATER invocation's glob-based aggregation can recover it without having
    # re-run this seed. See discover_seed_trust().
    sanity_df = pd.DataFrame([{"check": name, "passed": ok, "detail": detail}
                              for name, ok, detail in checks])
    sanity_df.insert(0, "seed", seed)
    sanity_df.to_csv(out_dir / f"dqn_sanity_seed{seed}.csv", index=False)

    # ---- model weights, seed-suffixed, never collides with the shipped model ----
    weights_path = weights_dir / f"dqn_weights_seed{seed}.npz"
    np.savez(weights_path,
             **{f"W{i}": w for i, w in enumerate(agent.online.W)},
             **{f"b{i}": b for i, b in enumerate(agent.online.b)})

    elapsed = time.time() - t0
    print(f"\n  Seed {seed} done in {elapsed / 60:.1f} min. Outputs:")
    for p in (seed_checkpoints if unsuffixed.exists() or seed_checkpoints.exists() else None,
             log_path, eval_path, weights_path):
        if p is not None:
            print(f"    {_display_path(p)}")

    dqn_row = eval_df.loc["DQN"].to_dict()
    dqn_row["seed"] = seed
    dqn_row["all_sanity_checks_passed"] = all_passed
    dqn_row["wall_time_sec"] = elapsed
    return dqn_row


def _seeds_from_filenames(paths) -> dict[int, Path]:
    """{seed_number: path}, parsed from every ...seed<N>.csv path given."""
    out = {}
    for p in paths:
        m = _SEED_NUM_RE.search(p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def discover_seed_evaluations(out_dir: Path) -> dict[int, dict]:
    """
    Every dqn_evaluation_seed*.csv actually present in out_dir, keyed by seed number, mapped
    to that seed's DQN row as a dict. This globs the DIRECTORY rather than trusting any
    in-memory list of seeds, which is what lets a later invocation's aggregation see seeds
    trained by an earlier invocation.
    """
    rows = {}
    for seed, path in sorted(_seeds_from_filenames(out_dir.glob("dqn_evaluation_seed*.csv")).items()):
        df = pd.read_csv(path, index_col=0)
        rows[seed] = df.loc["DQN"].to_dict()
    return rows


def discover_seed_trust(out_dir: Path) -> dict[int, bool]:
    """
    all_sanity_checks_passed per seed, by seed number, from whichever seeds have a record on
    disk - not from this invocation's memory. Two sources, sidecar taking precedence:

      1. dqn_sanity_seed<N>.csv - written by run_one_seed for every seed trained under this
         version of the script. Authoritative going forward.
      2. multiseed_per_seed_summary.csv, if present - covers seeds trained before the sidecar
         existed, so upgrading this script does not silently drop their trust status (and
         with it, their contribution to the aggregate) just because the bookkeeping changed.

    A seed with neither source is treated as UNTRUSTED (excluded, with a warning) rather
    than assumed passing - the earlier design's rule was "prove it passed", not "assume it
    did", and that should not weaken just because the record of it lives on disk instead of
    in memory.
    """
    trust: dict[int, bool] = {}
    for seed, path in sorted(_seeds_from_filenames(out_dir.glob("dqn_sanity_seed*.csv")).items()):
        trust[seed] = bool(pd.read_csv(path)["passed"].all())

    legacy_path = out_dir / "multiseed_per_seed_summary.csv"
    if legacy_path.exists():
        legacy = pd.read_csv(legacy_path)
        for _, row in legacy.iterrows():
            s = int(row["seed"])
            if s not in trust:  # sidecar, when present, always wins
                trust[s] = bool(row["all_sanity_checks_passed"])
    return trust


def aggregate(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    mean/std/min/max across EVERY seed present on disk in out_dir (see
    discover_seed_evaluations), restricted to seeds that passed every sanity check (see
    discover_seed_trust) - a seed with an illegal-action bug or a non-deterministic greedy
    policy is a broken measurement, not a data point on the variance being studied.

    Rebuilding from disk on every call - rather than from whatever seeds this process just
    trained - is what makes running additional seeds later EXTEND the aggregate instead of
    replacing it.
    """
    evaluations = discover_seed_evaluations(out_dir)
    trust = discover_seed_trust(out_dir)

    rows = []
    untrusted_seeds = []
    for seed, row in evaluations.items():
        passed = trust.get(seed)
        if passed is None:
            print(f"  WARNING: seed {seed} has an evaluation file but no sanity-check "
                  f"record (no sidecar, no legacy summary entry) - excluding it rather than "
                  f"assuming it passed.")
            passed = False
        row = dict(row)
        row["seed"] = seed
        row["all_sanity_checks_passed"] = passed
        rows.append(row)
        if not passed:
            untrusted_seeds.append(seed)

    df = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    trusted = df[df["all_sanity_checks_passed"]]
    if untrusted_seeds:
        print(f"\n  NOTE: seed(s) {untrusted_seeds} failed a sanity check (or have no "
              f"verifiable record) and are EXCLUDED from the aggregate stats below.")

    stats = trusted[HEADLINE_METRICS].agg(["mean", "std", "min", "max"]).T
    stats.insert(0, "n_seeds", len(trusted))
    stats.index.name = "metric"
    return stats.reset_index(), df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--reward", choices=sorted(REWARD_VARIANTS), default="default",
                        help="Reward variant to train and evaluate under. A variant other "
                             "than 'default' writes to its own subdirectory so it can never "
                             "overwrite another variant's results.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Override the output CSV directory. Advanced/testing use only "
                             "(e.g. verifying the aggregation logic against a scratch copy "
                             "without touching real results) - leave unset for a real run.")
    parser.add_argument("--weights-dir", type=Path, default=None,
                        help="Override the model weights directory. Same testing use as "
                             "--out-dir.")
    args = parser.parse_args()

    reward_fn = REWARD_VARIANTS[args.reward]
    # "default" keeps Step 1's exact existing layout (results/csv/multiseed/, models/) so
    # a default-reward re-run stays byte-path-compatible with everything already produced.
    # Any other variant gets its own subdirectory, which is the sole thing that guarantees
    # it can never collide with or overwrite another variant's files. --out-dir/--weights-dir
    # override both, for testing against an isolated directory.
    out_dir = args.out_dir or (MULTISEED_CSV_DIR if args.reward == "default"
                               else MULTISEED_CSV_DIR / args.reward)
    weights_dir = args.weights_dir or (MODEL_DIR if args.reward == "default"
                                       else MODEL_DIR / args.reward)

    ensure_output_dirs()
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print(f"MULTI-SEED VARIANCE STUDY: reward={args.reward}, {len(args.seeds)} seeds "
          f"{args.seeds}, {args.episodes} episodes each")
    print(f"Outputs -> {_display_path(out_dir)}/ and "
          f"{_display_path(weights_dir)}/")
    print("=" * 100)

    t_start = time.time()
    for s in args.seeds:
        run_one_seed(s, args.episodes, args.eval_every, args.reward, reward_fn,
                    out_dir, weights_dir)
    total_elapsed = time.time() - t_start

    # Rebuilt from EVERY dqn_evaluation_seed*.csv present in out_dir, not just the seeds
    # args.seeds names in this invocation - so running more seeds later extends this instead
    # of replacing it. See aggregate()'s docstring.
    stats, per_seed_df = aggregate(out_dir)
    all_seeds_on_disk = sorted(per_seed_df["seed"].tolist())
    newly_trained = sorted(args.seeds)

    per_seed_path = out_dir / "multiseed_per_seed_summary.csv"
    per_seed_df.to_csv(per_seed_path, index=False)

    stats_path = out_dir / "multiseed_aggregate_stats.csv"
    stats.to_csv(stats_path, index=False)

    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    print(f"  reward variant      : {args.reward}")
    print(f"  seeds trained THIS run : {newly_trained}")
    print(f"  seeds on disk (all)    : {all_seeds_on_disk}  <- aggregate below covers these")
    print(f"  episodes per seed      : {args.episodes}")
    print(f"  total wall time (this run) : {total_elapsed / 60:.1f} min")
    print()
    with pd.option_context("display.width", 140, "display.float_format", lambda v: f"{v:,.2f}"):
        print(f"  Per-seed DQN evaluation (ALL {len(per_seed_df)} seeds on disk):")
        print(per_seed_df[["seed"] + HEADLINE_METRICS + ["all_sanity_checks_passed"]]
              .to_string(index=False))
        print()
        print(f"  Aggregate (mean / std / min / max across {stats['n_seeds'].iloc[0]} "
              f"sanity-check-passing seeds):")
        print(stats.to_string(index=False))
    print()
    print(f"  Per-seed summary -> {_display_path(per_seed_path)}")
    print(f"  Aggregate stats   -> {_display_path(stats_path)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
