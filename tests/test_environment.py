"""
Automated correctness tests for the cluster scheduling environment and baselines.

Run directly for a PASS/FAIL summary table:
    python3 test_environment.py

Or under pytest (every check is also a plain `test_*` function):
    pytest test_environment.py -v

Checks:
    1. RESOURCE CONSERVATION   - per-machine allocation never exceeds capacity
    2. GOLD NEVER RELEASES     - Gold holds its resources from placement to episode end
    3. BRONZE RELEASES ON TIME - Bronze frees at exactly scheduled_time + duration
    4. NO-OP CORRECTNESS       - an unplaceable job stays waiting and unscheduled
    5. JOB CONSERVATION        - scheduled + waiting + not_arrived == total, every step
    6. EPISODE TERMINATION     - a random-action episode terminates within its step bound
    7. BASELINE ORDERING       - StaticPriority Gold SLA violation rate <= FCFS's
    8. RESERVATION TRADE-OFF   - Reservation Bronze waiting time >= StaticPriority's

SCALE: the step-level invariant checks (1, 2, 3, 5) run on a documented SUBSET of the real
trace - 300 Gold + 1200 Bronze on 30 machines - because they snapshot the full cluster
state on every step and the full 8,830-job episode would take minutes per check. The subset
is real trace data, and 30 machines against 300 Gold keeps the cluster genuinely contended
so the no-op and capacity paths are actually exercised. Checks 6, 7 and 8 run on the FULL
dataset, where the numbers that matter for the project are produced.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

# The modules under test live in src/; add it to the path so this file runs from anywhere
# (project root, tests/, or under pytest).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from baselines import (FCFSScheduler, RandomPolicy, ResourceReservationScheduler,
                       RoundRobinScheduler, StaticPriorityScheduler, run_episode)
from environment import (ClusterSchedulingEnv, default_machine_subset, load_machine_capacities,
                         JOBS_FILE)

TOL = 1e-9

_JOBS_CACHE: pd.DataFrame | None = None


def _jobs() -> pd.DataFrame:
    global _JOBS_CACHE
    if _JOBS_CACHE is None:
        _JOBS_CACHE = pd.read_csv(JOBS_FILE)
    return _JOBS_CACHE


def make_small_env(n_gold: int = 300, n_bronze: int = 1200, n_machines: int = 30,
                   **kwargs) -> ClusterSchedulingEnv:
    """A contended slice of the real trace, small enough for per-step instrumentation."""
    jobs = _jobs()
    gold = jobs[jobs["tier"] == "Gold"].head(n_gold)
    bronze = jobs[jobs["tier"] == "Bronze"].head(n_bronze)
    subset = pd.concat([gold, bronze]).sort_values("arrival_time").reset_index(drop=True)

    machines = load_machine_capacities(machine_ids=default_machine_subset(),
                                       n_machines=n_machines)
    return ClusterSchedulingEnv(jobs=subset, machines=machines, **kwargs)


def make_full_env(**kwargs) -> ClusterSchedulingEnv:
    """The full 8,830-job / 150-machine environment."""
    return ClusterSchedulingEnv(jobs=_jobs(), **kwargs)


class StateRecorder:
    """
    Step hook that tracks, per job, when its resources became and stopped being allocated,
    plus the running per-step invariant violations.

    Allocation transitions are detected by diffing the set of running jobs between steps and
    stamping the transition with the environment clock at that step.
    """

    def __init__(self, env: ClusterSchedulingEnv):
        self.env = env
        self.capacity_violations: list[str] = []
        self.conservation_violations: list[str] = []
        self.alloc_start: dict[int, float] = {}
        self.alloc_end: dict[int, float] = {}
        self._prev_running: set[int] = set()
        self.n_steps = 0
        self.final_time = 0.0

    def __call__(self, env: ClusterSchedulingEnv, action: int, info: dict) -> None:
        self.n_steps += 1
        self.final_time = env.now

        # --- check 1: no machine over capacity -------------------------------------
        over_cpu = np.flatnonzero(env.alloc_cpu > env.cpu_capacity + TOL)
        over_mem = np.flatnonzero(env.alloc_mem > env.mem_capacity + TOL)
        for m in over_cpu:
            self.capacity_violations.append(
                f"step {env.steps} t={env.now}: machine {m} cpu "
                f"{env.alloc_cpu[m]:.6f} > capacity {env.cpu_capacity[m]:.6f}")
        for m in over_mem:
            self.capacity_violations.append(
                f"step {env.steps} t={env.now}: machine {m} mem "
                f"{env.alloc_mem[m]:.6f} > capacity {env.mem_capacity[m]:.6f}")

        # --- check 5: job conservation ---------------------------------------------
        total = len(env.scheduled) + len(env.waiting) + (env.n_jobs - env._next_arrival_ptr)
        if total != env.n_jobs:
            self.conservation_violations.append(
                f"step {env.steps} t={env.now}: scheduled {len(env.scheduled)} + waiting "
                f"{len(env.waiting)} + not_arrived {env.n_jobs - env._next_arrival_ptr} "
                f"= {total} != {env.n_jobs}")
        # A job must never appear in two places at once.
        overlap = set(env.waiting) & set(env.scheduled)
        if overlap:
            self.conservation_violations.append(
                f"step {env.steps}: {len(overlap)} jobs both waiting and scheduled")

        # --- allocation transitions (checks 2 and 3) --------------------------------
        running = set(env.running)
        for idx in running - self._prev_running:
            self.alloc_start[idx] = env.jobs[idx].scheduled_time
        for idx in self._prev_running - running:
            self.alloc_end[idx] = env.now
        self._prev_running = running


# ----------------------------------------------------------------------------------
# CHECKS
# ----------------------------------------------------------------------------------

def check_1_resource_conservation() -> str:
    """Allocated CPU/mem per machine never exceeds that machine's capacity."""
    env = make_small_env()
    rec = StateRecorder(env)
    run_episode(env, RandomPolicy(seed=1), seed=1, step_hook=rec)

    assert not rec.capacity_violations, (
        f"{len(rec.capacity_violations)} capacity violations across {rec.n_steps} steps. "
        f"First 5:\n  " + "\n  ".join(rec.capacity_violations[:5]))

    assert (env.alloc_cpu >= -TOL).all(), "negative CPU allocation (over-release)"
    assert (env.alloc_mem >= -TOL).all(), "negative memory allocation (over-release)"
    return (f"{rec.n_steps} steps checked, {env.n_machines} machines, "
            f"0 capacity violations; peak cpu "
            f"{(env.alloc_cpu / env.cpu_capacity).max():.1%}")


def check_2_gold_never_releases() -> str:
    """Every scheduled Gold job stays allocated from its placement to episode end."""
    env = make_small_env()
    rec = StateRecorder(env)
    run_episode(env, RandomPolicy(seed=2), seed=2, step_hook=rec)

    gold_scheduled = [j for j in env.jobs if j.is_gold and j.scheduled_time is not None]
    assert gold_scheduled, "no Gold jobs were scheduled - test is vacuous"

    # released_at is stamped inside _release_completions, so this catches a release at the
    # instant it happens rather than inferring it from a step-boundary snapshot.
    released = [j for j in gold_scheduled if j.released_at is not None]
    assert not released, (
        f"{len(released)} Gold jobs released resources before episode end. First 5: " +
        ", ".join(f"{j.job_id}@t={j.released_at}" for j in released[:5]))

    # Cross-check against the independent step-boundary observation.
    observed = [j for j in gold_scheduled if j.index in rec.alloc_end]
    assert not observed, (
        f"{len(observed)} Gold jobs left the running set mid-episode. First 5: " +
        ", ".join(f"{j.job_id}@t={rec.alloc_end[j.index]}" for j in observed[:5]))

    # And they must still be present in the live running set at the end.
    still_running = set(env.running)
    missing = [j.job_id for j in gold_scheduled if j.index not in still_running]
    assert not missing, (
        f"{len(missing)} Gold jobs missing from the running set at episode end: "
        f"{missing[:5]}")

    # Their resources must still be reflected in the machine allocations.
    expected_cpu = np.zeros(env.n_machines)
    for j in gold_scheduled:
        expected_cpu[j.machine] += j.cpu_demand
    assert (env.alloc_cpu + TOL >= expected_cpu).all(), (
        "machine CPU allocation is lower than the Gold jobs pinned to it")

    return (f"{len(gold_scheduled)} Gold jobs scheduled, 0 released, all still allocated "
            f"at t={env.now:,.0f}s")


def check_3_bronze_releases_on_time() -> str:
    """
    Every scheduled Bronze job frees its resources at exactly scheduled_time + duration.

    OBSERVATION POINT: this asserts on `job.released_at`, stamped inside
    _release_completions at the instant the resources are handed back. An earlier version
    of this check diffed the running set at step boundaries instead, and reported the 6
    zero-duration Bronze jobs as one second late - _advance_clock settles the release at
    t=583 and only then moves the clock to 584, so the step-boundary snapshot sees the
    already-completed release stamped with the new clock. The environment was correct; the
    observation point was wrong. Reading the release time where the release happens
    removes the ambiguity entirely.
    """
    env = make_small_env()
    rec = StateRecorder(env)
    run_episode(env, RandomPolicy(seed=3), seed=3, step_hook=rec)

    bronze_scheduled = [j for j in env.jobs if not j.is_gold and j.scheduled_time is not None]
    assert bronze_scheduled, "no Bronze jobs were scheduled - test is vacuous"

    early, late, never = [], [], []
    checked = 0
    for j in bronze_scheduled:
        expected = j.scheduled_time + j.duration
        observed = j.released_at
        if observed is None:
            # Legitimate only if the episode ended before the job was due to finish.
            if expected <= env.now + TOL:
                never.append(f"{j.job_id} due t={expected} but never released "
                             f"(episode ended t={env.now})")
            continue
        checked += 1
        if observed < expected - TOL:
            early.append(f"{j.job_id}: released t={observed}, expected t={expected} "
                         f"(scheduled {j.scheduled_time} + duration {j.duration})")
        elif observed > expected + TOL:
            late.append(f"{j.job_id}: released t={observed}, expected t={expected} "
                        f"(scheduled {j.scheduled_time} + duration {j.duration})")

    problems = early + late + never
    assert not problems, (
        f"{len(early)} early, {len(late)} late, {len(never)} never released, "
        f"out of {len(bronze_scheduled)} Bronze jobs. First 5:\n  " +
        "\n  ".join(problems[:5]))
    assert checked > 0, "no Bronze release was actually observed - test is vacuous"

    return (f"{checked} Bronze releases verified to the exact second "
            f"({len(bronze_scheduled) - checked} still running at episode end)")


def check_4_noop_correctness() -> str:
    """
    A job that fits nowhere must stay waiting and must NOT be marked scheduled.

    Built as a synthetic 1-machine cluster whose capacity is deliberately smaller than the
    single job's demand, so the failure path is guaranteed rather than hoped for.
    """
    jobs = pd.DataFrame([
        {"job_id": "too_big", "tier": "Gold", "arrival_time": 0, "duration": float("nan"),
         "cpu_demand": 999.0, "mem_demand": 0.5, "continuous": True},
    ])
    machines = pd.DataFrame([{"machine_id": 1, "capacity_cpu": 8.0, "capacity_mem": 1.0}])
    env = ClusterSchedulingEnv(jobs=jobs, machines=machines)

    assert env.waiting == [0], "the oversized job should be waiting after reset"
    assert env.can_place(env.jobs[0]) is False, "job should not fit anywhere"

    obs, reward, terminated, truncated, info = env.step(0)  # try to place it

    assert info["scheduled"] is False, "environment reported a placement that cannot fit"
    assert 0 in env.waiting, "job left the waiting list despite not being placed"
    assert 0 not in env.scheduled, "job was marked scheduled despite not being placed"
    assert env.jobs[0].scheduled_time is None, "scheduled_time set on an unplaced job"
    assert env.jobs[0].machine is None, "machine assigned to an unplaced job"
    assert env.alloc_cpu.sum() == 0.0, "resources allocated for an unplaced job"
    assert env.alloc_mem.sum() == 0.0, "resources allocated for an unplaced job"
    assert reward < 0, f"expected a penalty for the wasted action, got reward {reward}"

    # The same must hold on the real trace: count no-ops that changed nothing.
    real = make_small_env()
    real.reset(seed=4)
    before = (len(real.scheduled), real.alloc_cpu.sum())
    real.step(real.k)  # explicit no-op action
    after = (len(real.scheduled), real.alloc_cpu.sum())
    assert before == after, "explicit no-op changed the allocation state"

    return ("oversized job stayed waiting, unscheduled, zero resources allocated, "
            f"penalty reward={reward:.4f}; explicit no-op is state-preserving")


def check_5_job_conservation() -> str:
    """scheduled + waiting + not_yet_arrived == total_jobs, at every single step."""
    env = make_small_env()
    rec = StateRecorder(env)
    run_episode(env, RandomPolicy(seed=5), seed=5, step_hook=rec)

    assert not rec.conservation_violations, (
        f"{len(rec.conservation_violations)} conservation violations. First 5:\n  " +
        "\n  ".join(rec.conservation_violations[:5]))

    # Final accounting, independently of the per-step hook.
    total = len(env.scheduled) + len(env.waiting) + (env.n_jobs - env._next_arrival_ptr)
    assert total == env.n_jobs, f"final accounting {total} != {env.n_jobs}"
    assert len(set(env.scheduled)) == len(env.scheduled), "a job was scheduled twice"
    assert len(set(env.waiting)) == len(env.waiting), "a job appears twice in the queue"
    assert len(env.running) + len(env.completed) == len(env.scheduled), (
        f"running {len(env.running)} + completed {len(env.completed)} "
        f"!= scheduled {len(env.scheduled)}")

    return (f"{rec.n_steps} steps, {env.n_jobs} jobs, 0 violations; "
            f"final: {len(env.scheduled)} scheduled + {len(env.waiting)} waiting")


def check_6_episode_termination() -> str:
    """A random-action episode terminates within the environment's step bound."""
    env = make_full_env()
    policy = RandomPolicy(seed=6)
    env.reset(seed=6)

    terminated = truncated = False
    steps = 0
    for _ in range(env.max_steps + 10):
        _, _, terminated, truncated, _ = env.step(policy(env))
        steps += 1
        if terminated or truncated:
            break

    assert terminated or truncated, (
        f"episode did not finish within {env.max_steps} steps (clock at {env.now})")
    # Ending at the horizon is a legitimate time-limit truncation (and is now reported as
    # truncated rather than terminated). Exhausting the STEP cap is the real failure - it
    # means the simulation was spinning without the clock making progress.
    assert steps < env.max_steps, (
        f"episode exhausted the {env.max_steps} step cap - clock only reached "
        f"{env.now}/{env.horizon} with {len(env.waiting)} still waiting")
    assert steps <= env.max_steps, f"{steps} steps exceeds bound {env.max_steps}"
    assert env.now <= env.horizon + TOL, f"clock {env.now} ran past horizon {env.horizon}"

    how = "completed all jobs" if terminated else "reached the time horizon"
    return (f"{how} after {steps:,} steps (cap {env.max_steps:,}), "
            f"sim clock {env.now:,.0f}s")


def _per_job_delays(env: ClusterSchedulingEnv) -> dict[str, float]:
    return {j.job_id: j.scheduling_delay(env.now) for j in env.jobs if j.is_gold}


_BASELINE_CACHE: dict[str, tuple[dict, ClusterSchedulingEnv]] = {}


def _run_baseline(policy) -> tuple[dict, ClusterSchedulingEnv]:
    """Run a baseline on the full environment, caching so checks 7 and 8 share the work."""
    name = policy.name
    if name not in _BASELINE_CACHE:
        env = make_full_env()
        metrics = run_episode(env, policy, seed=0)
        _BASELINE_CACHE[name] = (metrics, env)
    return _BASELINE_CACHE[name]


def check_7_baseline_ordering() -> str:
    """
    Static Priority must never protect Gold worse than priority-blind FCFS.

    On failure this prints per-job scheduling delays for both policies rather than just
    reporting the aggregate.
    """
    fcfs_m, fcfs_env = _run_baseline(FCFSScheduler())
    sp_m, sp_env = _run_baseline(StaticPriorityScheduler())

    sp_rate = sp_m["gold_sla_violation_rate"]
    fcfs_rate = fcfs_m["gold_sla_violation_rate"]

    if sp_rate > fcfs_rate:
        # Full diagnostics, as required, before the assertion fires.
        sp_delays = _per_job_delays(sp_env)
        fcfs_delays = _per_job_delays(fcfs_env)
        print("\n" + "=" * 78)
        print("CHECK 7 DIAGNOSTIC - StaticPriority protected Gold WORSE than FCFS")
        print("=" * 78)
        print(f"  StaticPriority Gold SLA violation rate: {sp_rate:.6f}")
        print(f"  FCFS           Gold SLA violation rate: {fcfs_rate:.6f}")
        print(f"  SLA threshold: {sp_env.sla_threshold}s\n")
        diffs = sorted(((sp_delays[k] - fcfs_delays[k], k) for k in sp_delays),
                       reverse=True)
        print(f"  {'job_id':24s} {'StaticPriority':>16s} {'FCFS':>12s} {'delta':>12s}")
        for delta, job_id in diffs[:40]:
            print(f"  {job_id:24s} {sp_delays[job_id]:16.2f} "
                  f"{fcfs_delays[job_id]:12.2f} {delta:12.2f}")
        print(f"\n  ({len(diffs)} Gold jobs total; showing the 40 largest regressions)")
        print(f"  StaticPriority scheduled {sp_m['n_gold_scheduled']} Gold, "
              f"FCFS scheduled {fcfs_m['n_gold_scheduled']}")
        print("=" * 78 + "\n")

    assert sp_rate <= fcfs_rate + TOL, (
        f"StaticPriority Gold SLA violation rate {sp_rate:.6f} > FCFS {fcfs_rate:.6f} "
        f"- see the diagnostic table above")

    return (f"StaticPriority {sp_rate:.4f} <= FCFS {fcfs_rate:.4f} "
            f"(Gold scheduled: {sp_m['n_gold_scheduled']} vs {fcfs_m['n_gold_scheduled']})")


def check_8_reservation_tradeoff() -> str:
    """Reserving capacity for Gold must cost Bronze something, not be free."""
    res_m, _ = _run_baseline(ResourceReservationScheduler(reserved_fraction=0.5))
    sp_m, _ = _run_baseline(StaticPriorityScheduler())

    res_wait = res_m["bronze_avg_waiting_time"]
    sp_wait = sp_m["bronze_avg_waiting_time"]

    assert res_wait >= sp_wait - TOL, (
        f"Resource Reservation Bronze average waiting time {res_wait:.4f}s is LOWER than "
        f"StaticPriority's {sp_wait:.4f}s - reserving capacity for Gold appears to have "
        f"made Bronze faster, which means the reservation constraint is not binding as "
        f"implemented.\n"
        f"  Reservation: {res_m['n_bronze_scheduled']} Bronze scheduled, "
        f"cpu_util {res_m['cpu_utilization']:.4f}\n"
        f"  StaticPriority: {sp_m['n_bronze_scheduled']} Bronze scheduled, "
        f"cpu_util {sp_m['cpu_utilization']:.4f}")

    return (f"Reservation Bronze wait {res_wait:,.2f}s >= StaticPriority {sp_wait:,.2f}s "
            f"(Bronze scheduled: {res_m['n_bronze_scheduled']} vs "
            f"{sp_m['n_bronze_scheduled']})")


CHECKS = [
    ("1. RESOURCE CONSERVATION", check_1_resource_conservation),
    ("2. GOLD NEVER RELEASES", check_2_gold_never_releases),
    ("3. BRONZE RELEASES ON TIME", check_3_bronze_releases_on_time),
    ("4. NO-OP CORRECTNESS", check_4_noop_correctness),
    ("5. JOB CONSERVATION", check_5_job_conservation),
    ("6. EPISODE TERMINATION", check_6_episode_termination),
    ("7. BASELINE ORDERING SANITY", check_7_baseline_ordering),
    ("8. RESERVATION TRADE-OFF", check_8_reservation_tradeoff),
]


# pytest entry points -----------------------------------------------------------------
def test_1_resource_conservation(): check_1_resource_conservation()
def test_2_gold_never_releases(): check_2_gold_never_releases()
def test_3_bronze_releases_on_time(): check_3_bronze_releases_on_time()
def test_4_noop_correctness(): check_4_noop_correctness()
def test_5_job_conservation(): check_5_job_conservation()
def test_6_episode_termination(): check_6_episode_termination()
def test_7_baseline_ordering(): check_7_baseline_ordering()
def test_8_reservation_tradeoff(): check_8_reservation_tradeoff()


def main() -> int:
    import time

    print("=" * 100)
    print("CLUSTER SCHEDULING ENVIRONMENT - AUTOMATED CORRECTNESS TESTS")
    print("=" * 100)

    results = []
    for name, fn in CHECKS:
        print(f"\n>>> {name} ... ", end="", flush=True)
        t0 = time.time()
        try:
            detail = fn()
            elapsed = time.time() - t0
            print(f"PASS ({elapsed:.1f}s)")
            print(f"    {detail}")
            results.append((name, "PASS", detail, elapsed))
        except AssertionError as exc:
            elapsed = time.time() - t0
            print(f"FAIL ({elapsed:.1f}s)")
            print(f"    {exc}")
            results.append((name, "FAIL", str(exc).split(chr(10))[0], elapsed))
        except Exception as exc:  # noqa: BLE001 - surface anything unexpected, don't hide it
            elapsed = time.time() - t0
            print(f"ERROR ({elapsed:.1f}s)")
            traceback.print_exc()
            results.append((name, "ERROR", f"{type(exc).__name__}: {exc}", elapsed))

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"{'CHECK':<32s} {'RESULT':<8s} {'TIME':>7s}   DETAIL")
    print("-" * 100)
    for name, status, detail, elapsed in results:
        mark = "PASS" if status == "PASS" else status
        print(f"{name:<32s} {mark:<8s} {elapsed:6.1f}s   {detail[:70]}")
    print("-" * 100)

    n_pass = sum(1 for r in results if r[1] == "PASS")
    print(f"{n_pass}/{len(results)} checks passed")
    print("=" * 100)

    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
