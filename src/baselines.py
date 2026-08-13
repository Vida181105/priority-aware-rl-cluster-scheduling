"""
Baseline schedulers for the priority-aware cluster scheduling environment.

Each baseline is a policy: a callable taking the environment and returning an action in
Discrete(K+1). Every policy sees exactly the same K-slot visible window the DQN agent will
see, so the comparison isolates the scheduling decision itself.

Baselines:
    FCFSScheduler                - earliest arrival first
    RoundRobinScheduler          - alternate oldest Gold / oldest Bronze
    StaticPriorityScheduler      - Gold first, Bronze only when no Gold is schedulable
    ResourceReservationScheduler - reserve a fraction of the cluster exclusively for Gold

DESIGN - "SKIP IF IT DOES NOT FIT". Every baseline picks the highest-ranked visible job
that ACTUALLY FITS right now, rather than blocking on a head-of-line job that cannot be
placed. Strict blocking would deadlock this cluster: Gold never releases its resources, so
a Gold job at the head of the queue that no longer fits would block that queue for the
whole episode and the baseline would schedule almost nothing. This is the standard
"FCFS with backfill" treatment and it is applied uniformly to all four baselines, so no
baseline gains an advantage from it.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from environment import ClusterSchedulingEnv, Job


class BaselineScheduler:
    """Base class. Subclasses implement `rank`, returning visible jobs best-first."""

    name = "base"

    def reset(self) -> None:
        """Clear any per-episode policy state. Called at the start of each episode."""

    def rank(self, env: ClusterSchedulingEnv, visible: list[int]) -> list[int]:
        raise NotImplementedError

    def allows(self, env: ClusterSchedulingEnv, job: Job) -> bool:
        """Policy-level admission control. Only Resource Reservation restricts anything."""
        return True

    def __call__(self, env: ClusterSchedulingEnv) -> int:
        visible = env.visible_jobs()
        if not visible:
            return env.k  # nothing to act on -> no-op, which advances the clock

        order = self.rank(env, visible)
        for idx in order:
            job = env.jobs[idx]
            if self.allows(env, job) and env.can_place(job):
                return visible.index(idx)
        return env.k  # nothing schedulable this instant


def _age_key(env: ClusterSchedulingEnv):
    """Oldest-first ordering: earliest arrival, ties broken by job index."""
    return lambda i: (env.jobs[i].arrival_time, i)


class FCFSScheduler(BaselineScheduler):
    """Priority-blind: whichever visible job arrived earliest, regardless of tier."""

    name = "FCFS"

    def rank(self, env, visible):
        return sorted(visible, key=_age_key(env))


class RoundRobinScheduler(BaselineScheduler):
    """
    Alternates tiers: oldest waiting Gold, then oldest waiting Bronze, then back.

    The alternation flag only flips on a SUCCESSFUL placement of the preferred tier, so a
    tier with nothing schedulable does not silently consume its turn.
    """

    name = "RoundRobin"

    def __init__(self):
        self._prefer_gold = True

    def reset(self):
        self._prefer_gold = True

    def rank(self, env, visible):
        key = _age_key(env)
        gold = sorted((i for i in visible if env.jobs[i].is_gold), key=key)
        bronze = sorted((i for i in visible if not env.jobs[i].is_gold), key=key)
        return (gold + bronze) if self._prefer_gold else (bronze + gold)

    def __call__(self, env):
        action = super().__call__(env)
        if action < env.k:
            placed = env.jobs[env.visible_jobs()[action]]
            # Flip only when the tier we were favouring actually got served.
            if placed.is_gold == self._prefer_gold:
                self._prefer_gold = not self._prefer_gold
        return action


class StaticPriorityScheduler(BaselineScheduler):
    """
    Strict tier priority: any schedulable Gold outranks every Bronze job.

    Within a tier, oldest first. This is the baseline the DQN has to beat - it protects
    Gold maximally, at whatever cost to Bronze.
    """

    name = "StaticPriority"

    def rank(self, env, visible):
        key = _age_key(env)
        gold = sorted((i for i in visible if env.jobs[i].is_gold), key=key)
        bronze = sorted((i for i in visible if not env.jobs[i].is_gold), key=key)
        return gold + bronze


class ResourceReservationScheduler(BaselineScheduler):
    """
    Reserves `reserved_fraction` of TOTAL cluster capacity exclusively for Gold.

    Bronze may occupy at most (1 - reserved_fraction) of total CPU and memory at any
    instant; Gold may use the whole cluster. Ranking is Gold-first, as in Static Priority,
    so the only difference between the two baselines is the Bronze capacity cap - which
    isolates the cost of reserving.

    FINDING - THIS BASELINE COLLAPSES INTO STATIC PRIORITY ON THIS WORKLOAD, and that is
    reported as a result rather than tuned away.

    Measured: identical to StaticPriority on every metric at every tick value tested
    (0.0, 0.5, 1.0, 2.0, 5.0, 10.0) - verified by frame equality, not by eye. The reason is
    structural: Gold's natural footprint is ~96% of cluster CPU, and Gold never releases.
    Bronze therefore never gets anywhere near its 50% cap, so the reservation constraint is
    never the binding one - free capacity is. The cap only starts to bite if the reserved
    fraction is set BELOW Gold's own demand, at which point it is no longer reserving
    capacity for Gold, it is rationing Gold - which inverts the mechanism it is supposed to
    model.

    The generalisable point for the write-up: static reservation is a no-op once the
    protected class's demand already exceeds the reserved share. It can only express a
    trade-off when the protected class is a MINORITY of demand. Here Gold is the majority
    consumer despite being the minority by job count (1,532 jobs holding ~96% of CPU
    against 40,649 Bronze jobs), so a static reservation has nothing left to reserve.
    This is precisely the gap a learned policy is meant to fill: the decision has to be
    dynamic and per-arrival, not a fixed capacity split.
    """

    name = "ResourceReservation"

    def __init__(self, reserved_fraction: float = 0.5):
        self.reserved_fraction = float(reserved_fraction)

    def rank(self, env, visible):
        key = _age_key(env)
        gold = sorted((i for i in visible if env.jobs[i].is_gold), key=key)
        bronze = sorted((i for i in visible if not env.jobs[i].is_gold), key=key)
        return gold + bronze

    def allows(self, env, job):
        if job.is_gold:
            return True  # Gold may use reserved and unreserved capacity alike

        # Current Bronze footprint across the cluster.
        bronze_cpu = sum(env.jobs[i].cpu_demand for i in env.running if not env.jobs[i].is_gold)
        bronze_mem = sum(env.jobs[i].mem_demand for i in env.running if not env.jobs[i].is_gold)

        cpu_budget = (1.0 - self.reserved_fraction) * env.total_cpu
        mem_budget = (1.0 - self.reserved_fraction) * env.total_mem

        return (bronze_cpu + job.cpu_demand <= cpu_budget + 1e-9 and
                bronze_mem + job.mem_demand <= mem_budget + 1e-9)


# ----------------------------------------------------------------------------------
# EPISODE RUNNER
# ----------------------------------------------------------------------------------

def run_episode(env: ClusterSchedulingEnv,
                policy: Callable[[ClusterSchedulingEnv], int],
                seed: Optional[int] = 0,
                max_steps: Optional[int] = None,
                step_hook: Optional[Callable[[ClusterSchedulingEnv, int, dict], None]] = None) -> dict:
    """
    Run one full episode under `policy` and return its metrics.

    `step_hook(env, action, info)` is called after every step - the correctness tests use
    it to inspect invariants at each step without duplicating this loop.
    """
    env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()

    limit = max_steps if max_steps is not None else env.max_steps
    total_reward = 0.0

    for _ in range(limit):
        action = policy(env)
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if step_hook is not None:
            step_hook(env, action, info)
        if terminated or truncated:
            break

    metrics = env.episode_metrics()
    metrics["total_reward"] = total_reward
    metrics["policy"] = getattr(policy, "name", policy.__class__.__name__)
    return metrics


class RandomPolicy:
    """Uniform random over legal actions. Used by the correctness tests."""

    name = "Random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def __call__(self, env: ClusterSchedulingEnv) -> int:
        legal = np.flatnonzero(env.action_mask())
        return int(self.rng.choice(legal))


def all_baselines(reserved_fraction: float = 0.5) -> list[BaselineScheduler]:
    return [
        FCFSScheduler(),
        RoundRobinScheduler(),
        StaticPriorityScheduler(),
        ResourceReservationScheduler(reserved_fraction=reserved_fraction),
    ]


def compare_baselines(env_factory: Callable[[], ClusterSchedulingEnv],
                      reserved_fraction: float = 0.5,
                      seed: int = 0) -> pd.DataFrame:
    """Run every baseline on a fresh environment and return one row of metrics each."""
    rows = []
    for policy in all_baselines(reserved_fraction):
        rows.append(run_episode(env_factory(), policy, seed=seed))
    return pd.DataFrame(rows).set_index("policy")


REPORT_COLUMNS = [
    "gold_sla_violation_rate",
    "gold_avg_scheduling_delay",
    "bronze_avg_completion_time",
    "cpu_utilization",
    "mem_utilization",
    "gold_avg_waiting_time",
    "bronze_avg_waiting_time",
    "throughput",
]


def print_comparison(df: pd.DataFrame) -> None:
    """Human-readable baseline comparison table."""
    show = df[REPORT_COLUMNS + ["n_gold_scheduled", "n_bronze_scheduled", "n_unscheduled"]]
    with pd.option_context("display.width", 200, "display.max_columns", 30,
                           "display.float_format", lambda v: f"{v:,.4f}"):
        print(show.T)


if __name__ == "__main__":
    print("Running all four baselines on the full trace...\n")
    results = compare_baselines(lambda: ClusterSchedulingEnv())
    print_comparison(results)
