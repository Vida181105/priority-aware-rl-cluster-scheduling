"""
Sweep SCHEDULING_TICK_SECONDS and report how the four baselines respond.

SCHEDULING_TICK_SECONDS is the simulated time a single placement decision consumes. At 0
the scheduler places an unbounded number of jobs per instant, so nothing ever queues and
all four baselines collapse to identical numbers. This sweep asks at what tick value a
real queue forms and the baselines start to separate.

COHORT SPLIT: 1,512 of the 1,532 Gold jobs arrive at t=0, where arrival order and tier
priority agree, so FCFS and Static Priority make identical choices on them by construction.
The only jobs where the two policies CAN disagree are the 20 Gold jobs arriving late
(t = 34,400-42,500 s) against a Bronze queue that has been waiting since t = 5 s. Averaged
over all 1,532 Gold those 20 are diluted ~75:1, so they are reported separately.

Run:  python3 sweep_scheduling_tick.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from baselines import all_baselines, run_episode
from environment import CSV_DIR, ClusterSchedulingEnv, ensure_output_dirs

TICKS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]  # 0.0 included as the reference / regression row

METRICS = [
    "gold_sla_violation_rate",
    "gold_avg_scheduling_delay",
    "bronze_avg_waiting_time",
    "cpu_utilization",
    "n_bronze_scheduled",
]


def cohort_stats(env: ClusterSchedulingEnv) -> dict:
    """Split Gold into the t=0 block and the late arrivals, and measure each separately."""
    gold = [j for j in env.jobs if j.is_gold]
    early = [j for j in gold if j.arrival_time == 0]
    late = [j for j in gold if j.arrival_time > 0]

    def avg_delay(js):
        # Same convention as episode_metrics: unscheduled jobs are charged their censored
        # wait rather than dropped from the average.
        if not js:
            return float("nan")
        vals = [(j.scheduling_delay(env.now) if j.scheduled_time is not None
                 else env.now - j.arrival_time) for j in js]
        return float(np.mean(vals))

    return {
        "n_gold_late": len(late),
        "late_gold_avg_delay": avg_delay(late),
        "late_gold_scheduled": sum(1 for j in late if j.scheduled_time is not None),
        "t0_gold_avg_delay": avg_delay(early),
    }


def sweep(ticks=TICKS) -> pd.DataFrame:
    rows = []
    for tick in ticks:
        t0 = time.time()
        for policy in all_baselines():
            env = ClusterSchedulingEnv(scheduling_tick=tick)
            m = run_episode(env, policy, seed=0)
            m.update(cohort_stats(env))
            m["tick"] = tick
            rows.append(m)
        print(f"  tick={tick:<5} done in {time.time() - t0:5.1f}s", flush=True)
    return pd.DataFrame(rows)


ORDER = ["FCFS", "RoundRobin", "StaticPriority", "ResourceReservation"]


def main() -> None:
    print("Sweeping SCHEDULING_TICK_SECONDS over", TICKS)
    print("4 baselines x", len(TICKS), "tick values =", 4 * len(TICKS), "full episodes\n")

    df = sweep()

    fmt = lambda v: f"{v:,.4f}"  # noqa: E731
    with pd.option_context("display.width", 200, "display.max_columns", 40,
                           "display.float_format", fmt):

        print("\n\n" + "=" * 94)
        print("CROSS-TICK COMPARISON  (rows = tick, columns = baseline)")
        print("=" * 94)
        for metric in METRICS + ["n_gold_scheduled", "n_gold_unscheduled"]:
            print(f"\n--- {metric} ---")
            print(df.pivot(index="tick", columns="policy", values=metric)[ORDER])

        # ---- the whole point: the cohort where the policies can actually disagree -----
        print("\n\n" + "=" * 94)
        print("LATE-ARRIVING GOLD COHORT  (the 20 Gold jobs arriving t=34,400-42,500s)")
        print("These are the only Gold jobs where arrival order and tier priority disagree.")
        print("=" * 94)
        for metric in ["late_gold_avg_delay", "late_gold_scheduled"]:
            print(f"\n--- {metric} ---")
            print(df.pivot(index="tick", columns="policy", values=metric)[ORDER])

        print("\n--- t0_gold_avg_delay (the 1,512 jobs arriving at t=0, for contrast) ---")
        print(df.pivot(index="tick", columns="policy", values="t0_gold_avg_delay")[ORDER])

        # ---- does Static Priority beat FCFS? -----------------------------------------
        print("\n\n" + "=" * 94)
        print("DOES STATIC PRIORITY BEAT FCFS ON GOLD?")
        print("=" * 94)
        g = df.pivot(index="tick", columns="policy", values="gold_avg_scheduling_delay")
        lg = df.pivot(index="tick", columns="policy", values="late_gold_avg_delay")
        s = df.pivot(index="tick", columns="policy", values="gold_sla_violation_rate")
        print(pd.DataFrame({
            "all_gold_FCFS": g["FCFS"],
            "all_gold_StaticPri": g["StaticPriority"],
            "all_gold_improvement": g["FCFS"] - g["StaticPriority"],
            "LATE_gold_FCFS": lg["FCFS"],
            "LATE_gold_StaticPri": lg["StaticPriority"],
            "LATE_gold_improvement": lg["FCFS"] - lg["StaticPriority"],
            "sla_improvement": s["FCFS"] - s["StaticPriority"],
        }))
        print("\n(positive improvement = Static Priority protects Gold better than FCFS)")

    ensure_output_dirs()
    df.to_csv(CSV_DIR / "tick_sweep_results.csv", index=False)
    print(f"\nFull results written to {CSV_DIR / 'tick_sweep_results.csv'}")


if __name__ == "__main__":
    main()
