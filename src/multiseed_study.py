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

Run:
    python3 src/multiseed_study.py                                        # 5 seeds x 50 episodes, default reward
    python3 src/multiseed_study.py --seeds 0 1 --episodes 3               # smoke test
    python3 src/multiseed_study.py --reward freshness_bonus               # Step 2: same 5 seeds, freshness bonus
    python3 src/multiseed_study.py --reward freshness_bonus --seeds 0 1 --episodes 3  # Step 2 smoke test
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dqn_agent import EVAL_EVERY, DQNAgent, evaluate, sanity_checks, train
from environment import CSV_DIR, MODEL_DIR, default_reward, ensure_output_dirs, freshness_bonus_reward

MULTISEED_CSV_DIR = CSV_DIR / "multiseed"

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
        print(f"  moved {unsuffixed.name} -> {seed_checkpoints.relative_to(CSV_DIR.parent.parent)}")
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
            print(f"    {p.relative_to(CSV_DIR.parent.parent)}")

    dqn_row = eval_df.loc["DQN"].to_dict()
    dqn_row["seed"] = seed
    dqn_row["all_sanity_checks_passed"] = all_passed
    dqn_row["wall_time_sec"] = elapsed
    return dqn_row


def aggregate(seed_rows: list[dict]) -> pd.DataFrame:
    """
    mean/std/min/max across seeds for the headline metrics, computed ONLY from seeds that
    passed every sanity check - a seed with an illegal-action bug or a non-deterministic
    greedy policy is a broken measurement, not a data point on the variance being studied.
    """
    df = pd.DataFrame(seed_rows)
    trusted = df[df["all_sanity_checks_passed"]]
    dropped = len(df) - len(trusted)
    if dropped:
        print(f"\n  NOTE: {dropped} of {len(df)} seed(s) failed a sanity check and are "
              f"EXCLUDED from the aggregate stats below.")

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
    args = parser.parse_args()

    reward_fn = REWARD_VARIANTS[args.reward]
    # "default" keeps Step 1's exact existing layout (results/csv/multiseed/, models/) so
    # a default-reward re-run stays byte-path-compatible with everything already produced.
    # Any other variant gets its own subdirectory, which is the sole thing that guarantees
    # it can never collide with or overwrite another variant's files.
    out_dir = MULTISEED_CSV_DIR if args.reward == "default" else MULTISEED_CSV_DIR / args.reward
    weights_dir = MODEL_DIR if args.reward == "default" else MODEL_DIR / args.reward

    ensure_output_dirs()
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print(f"MULTI-SEED VARIANCE STUDY: reward={args.reward}, {len(args.seeds)} seeds "
          f"{args.seeds}, {args.episodes} episodes each")
    print(f"Outputs -> {out_dir.relative_to(CSV_DIR.parent.parent)}/ and "
          f"{weights_dir.relative_to(CSV_DIR.parent.parent)}/")
    print("=" * 100)

    t_start = time.time()
    seed_rows = [run_one_seed(s, args.episodes, args.eval_every, args.reward, reward_fn,
                              out_dir, weights_dir)
                for s in args.seeds]
    total_elapsed = time.time() - t_start

    stats, per_seed_df = aggregate(seed_rows)

    per_seed_path = out_dir / "multiseed_per_seed_summary.csv"
    per_seed_df.to_csv(per_seed_path, index=False)

    stats_path = out_dir / "multiseed_aggregate_stats.csv"
    stats.to_csv(stats_path, index=False)

    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    print(f"  reward variant      : {args.reward}")
    print(f"  seeds run           : {args.seeds}")
    print(f"  episodes per seed   : {args.episodes}")
    print(f"  total wall time     : {total_elapsed / 60:.1f} min")
    print()
    with pd.option_context("display.width", 140, "display.float_format", lambda v: f"{v:,.2f}"):
        print("  Per-seed DQN evaluation:")
        print(per_seed_df[["seed"] + HEADLINE_METRICS + ["all_sanity_checks_passed"]]
              .to_string(index=False))
        print()
        print("  Aggregate (mean / std / min / max across sanity-check-passing seeds):")
        print(stats.to_string(index=False))
    print()
    print(f"  Per-seed summary -> {per_seed_path.relative_to(CSV_DIR.parent.parent)}")
    print(f"  Aggregate stats   -> {stats_path.relative_to(CSV_DIR.parent.parent)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
