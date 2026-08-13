"""
Priority-aware cluster scheduling environment (Gymnasium).

Replays the cleaned Alibaba Cluster Trace v2017 job stream (data/jobs_clean.csv) as a
discrete-event scheduling simulation over a fixed set of machines whose capacities come
from data/server_event.csv.

Two tiers:
    Gold   (continuous=True)  - online service containers. Once scheduled they hold their
                                CPU/memory for the REST OF THE EPISODE. They never release.
    Bronze (continuous=False) - batch tasks. Release their resources at exactly
                                scheduled_time + duration.

The agent picks ONE waiting job per step from a bounded visible window; the machine is
chosen by the environment via first-fit (see _first_fit).

Design decisions are commented inline and marked with "DESIGN:".
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

# ----------------------------------------------------------------------------------
# CONFIGURATION CONSTANTS
# ----------------------------------------------------------------------------------

# PROJECT PATHS. Anchored on this file's location rather than the working directory, so
# every script resolves inputs and outputs identically whether it is launched from the
# project root, from src/, or from an editor.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CSV_DIR = RESULTS_DIR / "csv"
LOG_DIR = RESULTS_DIR / "logs"
PLOT_DIR = RESULTS_DIR / "plots"


def ensure_output_dirs() -> None:
    """Create the results/models directories if a script is about to write into them."""
    for d in (MODEL_DIR, CSV_DIR, LOG_DIR, PLOT_DIR):
        d.mkdir(parents=True, exist_ok=True)


JOBS_FILE = DATA_DIR / "jobs_clean.csv"
SERVER_FILE = DATA_DIR / "server_event.csv"

N_MACHINES = 150  # must match the preprocessing notebook's machine subset
K_VISIBLE = 10  # scheduler sees this many waiting jobs at a time

# MODELING ASSUMPTION: a Gold job whose scheduling delay exceeds this many seconds counts
# as an SLA violation. The trace carries no real SLA target, so 5 s is our own choice -
# it is a stand-in for "a latency-sensitive service should be placed almost immediately".
# Tune it and re-run; every reported violation rate depends on this number.
SLA_THRESHOLD_SECONDS = 5.0

# Penalty applied when the agent picks a job that cannot be placed anywhere, or no-ops
# while placeable work exists. Small, so it shapes behaviour without dominating the
# utilization term.
NOOP_PENALTY = 0.01
SLA_VIOLATION_PENALTY = 1.0

# MODELING ASSUMPTION: simulated seconds consumed per placement decision. At 0.0 the
# scheduler places an unlimited number of jobs per instant, so every job that fits on
# arrival gets a scheduling delay of exactly 0, nothing ever queues, and all four
# baselines produce byte-identical metrics - there is no scheduling problem to learn.
#
# LOCKED AT 1.0 after the tick sweep (see sweep_scheduling_tick.py / tick_sweep_results.csv).
# 1.0 s is the largest value at which the entire workload still completes inside the
# episode horizon - all 1,532 Gold and all 40,649 Bronze get scheduled - so no metric is
# confounded by horizon truncation, while the policies separate cleanly (on the 20
# late-arriving Gold jobs: FCFS 33.5 s average delay vs Static Priority 0.4 s). At 2.0 s
# and above, over half the Bronze workload never gets scheduled at all.
SCHEDULING_TICK_SECONDS = 1.0

# Weight on the Gold scheduling-delay term in the queue-integrated reward
# (queue_delay_reward, kept for comparison only). See default_reward for the current one.
GOLD_DELAY_WEIGHT = 20.0

# PRIORITY WEIGHTS on each tier's MEAN delay (see default_reward). Configurable; 10:1 in
# Gold's favour to start.
#
# These are applied to a per-tier COUNT-NORMALISED delay, not to raw per-job seconds. The
# first attempt weighted raw seconds 10:1 and the priority came out inverted, because
# weight-per-second is not weight-per-tier once the tiers differ in volume:
#     Gold   penalty ~ 0.010 * 1,532 jobs  * ~2,100 s  ~= -32,000
#     Bronze penalty ~ 0.001 * 40,649 jobs * ~8,060 s  ~= -328,000
# Bronze is 26.5x more numerous and waits ~4x longer, so it dominated the return by ~10:1
# and the agent's strongest incentive was to drain the Bronze queue - the exact opposite of
# a priority-aware scheduler. Dividing each placement by its tier's job count makes the
# episode sum telescope into the tier's MEAN delay, so the weights below really do express
# "one second of average Gold delay costs 10x one second of average Bronze delay",
# independent of how many jobs each tier happens to contain.
GOLD_PRIORITY_WEIGHT = 10.0
BRONZE_PRIORITY_WEIGHT = 1.0

# Weight on the utilization term. Cut from 1.0 to 0.05 because at 1.0 it contributed about
# +35,000 per episode - larger in magnitude than the Gold term - while varying almost not at
# all (every policy, learned or hand-written, lands at 0.9599 CPU utilization). That made it
# a large near-constant offset: after reward scaling it added ~+0.096 on nearly every one of
# 43,600 steps, so the value function converged toward a flat Q of roughly +9.6 everywhere
# and the discriminating signal - occasional large penalties for placing Gold late - became
# sparse noise on top of a constant. Utilization is still reported in episode_metrics().
UTIL_WEIGHT = 0.05

# CONTINUOUS QUEUE-HOLDING COST - now the SOLE delay penalty, hence 1.0.
#
# Per-placement attribution alone has excellent credit assignment but broken incentives:
# placing a Gold job that has waited 1,000 s costs -10 * 1000/1532 = -6.5, while a no-op
# costs -0.01 and leaving the job queued costs nothing at all. With a ~20-step effective
# horizon the agent cannot see that deferring makes the eventual bill larger, so stalling
# looks like a 650x saving - and it learned exactly that: over 20 episodes steps per episode
# fell 44,000 -> 37,360 while Bronze waiting nearly tripled, 3,617 s -> 9,880 s.
#
# Charging the queue continuously restores "waiting is never free" while keeping the
# placement term dominant for credit assignment. Count-normalised the same way, so the
# episode total is QUEUE_HOLD_FRACTION * (w_gold * mean_gold_delay + w_bronze *
# mean_bronze_delay) - i.e. exactly this fraction of the placement term.
QUEUE_HOLD_FRACTION = 1.0

# FRESHNESS BONUS - TRIED AND REVERTED. Not part of the shipped reward; kept here with its
# measured outcome because the negative result is worth reporting, and reproducible via
# freshness_bonus_reward() below.
#
# The idea: a bonus for placing a Gold job that has NOT yet blown its SLA, decaying with
# how long it has already waited:
#
#     bonus = GOLD_FRESHNESS_BONUS * exp(-wait / GOLD_FRESHNESS_TAU)
#
# Why this is needed. The potential-shaping credit is proportional to the wait a placement
# clears, so placing a job from the t=0 backlog (wait ~1,000 s) earns ~6.5 while placing a
# just-arrived Gold job (wait ~0 s) earns ~0. That is backwards for priority-awareness: the
# 20 late-arriving Gold jobs are precisely the ones a priority-aware scheduler must grab
# immediately, and the reward was paying the agent to prefer stale backlog work instead.
# Measured consequence: late-Gold delay of 79.4 s against Static Priority's 0.4 s, even
# once aggregate Gold delay had reached parity (757.7 vs 745.6).
#
# Decaying over the SLA threshold makes the bonus mean something concrete - "this job can
# still meet its SLA, place it now".
#
# OUTCOME (20-episode run, checkpoints every 5): no detectable improvement on the metric it
# targeted, and a worse tail. Best checkpoint reached late-Gold 101.2 s versus 79.4 s for the
# best checkpoint WITHOUT the bonus. Episode-matched it was better at ep 10 (627.7 vs 1,129.8)
# and far worse at ep 20 (1,409.0 vs 51.0) - i.e. noise. Episode 5 collapsed to 1,719 of
# 42,181 jobs placed, the worst in the project, consistent with (though not proven to be) the
# bonus making non-Gold work look worthless: at 10.0 against a Bronze credit of ~0.025 it is a
# 400x cliff, and after t=0 there are only 20 Gold arrivals in a 12-hour episode to wait for.
#
# The real blocker it exposed: checkpoint-to-checkpoint variance in late-Gold delay spans
# 51-2,294 s without the bonus and 101-9,560 s with it. Any reward change smaller than an
# order of magnitude is undetectable against that, so this question cannot be settled without
# a multi-seed study reporting mean +/- std rather than single checkpoints.
GOLD_FRESHNESS_BONUS = 10.0
GOLD_FRESHNESS_TAU = SLA_THRESHOLD_SECONDS

# Episode ends once the clock passes max(arrival_time) + this buffer.
EPISODE_BUFFER_SECONDS = 3600.0

SERVER_COLS = ["timestamp", "machineID", "event_type", "event_detail",
               "capacity_cpu", "capacity_mem", "capacity_disk"]


# ----------------------------------------------------------------------------------
# MACHINE CAPACITIES
# ----------------------------------------------------------------------------------

def load_machine_capacities(server_file: Path = SERVER_FILE,
                            machine_ids: Optional[np.ndarray] = None,
                            n_machines: int = N_MACHINES,
                            mem_mode: str = "rescaled") -> pd.DataFrame:
    """
    Load per-machine capacity from server_event.csv ("add" events only).

    DESIGN - MEMORY UNIT MISMATCH (important, and verified against the trace):
    `server_event.capacity_mem` is ~0.69 for nearly every machine, but the job table's
    `mem_demand` is normalised as a fraction of *that machine's own* RAM. The two columns
    are on different scales. Taking capacity_mem at face value puts 91 of the 150 machines
    OVER capacity from the real trace assignment alone (max 1.35x) - i.e. it would declare
    the actual, observed Alibaba placement infeasible, which cannot be right.

    Dividing capacity_mem by its fleet-wide median (~0.69) puts a typical machine at
    exactly 1.0, matching the jobs' normalisation, and every machine's real occupancy then
    fits with headroom (max 0.933, zero violations). Machine-to-machine heterogeneity from
    server_event is preserved by the rescale rather than thrown away.

    mem_mode:
        "rescaled" (default) - capacity_mem / median(capacity_mem). Recommended.
        "raw"                - use capacity_mem verbatim. Reproduces the infeasibility
                               above; kept so the mismatch can be demonstrated.
        "unit"               - every machine gets exactly 1.0.

    CPU needs no such treatment: capacity_cpu is a raw core count (64) and the job table's
    cpu_demand is also in cores.
    """
    df = pd.read_csv(server_file, header=None, names=SERVER_COLS)
    add = df[df["event_type"] == "add"].drop_duplicates("machineID", keep="first")

    if machine_ids is not None:
        add = add[add["machineID"].isin(machine_ids)]
    add = add.sort_values("machineID").head(n_machines).reset_index(drop=True)

    if len(add) < n_machines:
        raise ValueError(f"only {len(add)} machines found in {server_file}, need {n_machines}")

    caps = pd.DataFrame({
        "machine_id": add["machineID"].to_numpy(),
        "capacity_cpu": add["capacity_cpu"].astype(float).to_numpy(),
    })

    raw_mem = add["capacity_mem"].astype(float).to_numpy()
    if mem_mode == "rescaled":
        scale = float(np.median(df.loc[df["event_type"] == "add", "capacity_mem"]))
        caps["capacity_mem"] = raw_mem / scale
    elif mem_mode == "raw":
        caps["capacity_mem"] = raw_mem
    elif mem_mode == "unit":
        caps["capacity_mem"] = 1.0
    else:
        raise ValueError(f"unknown mem_mode {mem_mode!r}")

    return caps


def default_machine_subset(container_file: Optional[Path] = None,
                           n_machines: int = N_MACHINES) -> Optional[np.ndarray]:
    """
    The same machine subset the preprocessing notebook used: first N unique machine_ids in
    file order from container_event.csv. Returns None if that file is unavailable, in
    which case the caller falls back to the first N machines in server_event.csv.
    """
    container_file = container_file or (DATA_DIR / "container_event.csv")
    if not Path(container_file).exists():
        return None
    ce = pd.read_csv(container_file, header=None, index_col=False, usecols=range(8),
                     names=["ts", "event", "instance_id", "machine_id", "plan_cpu",
                            "plan_mem", "plan_disk", "cpuset"])
    return ce["machine_id"].drop_duplicates().head(n_machines).to_numpy()


# ----------------------------------------------------------------------------------
# JOB BOOKKEEPING
# ----------------------------------------------------------------------------------

@dataclass
class Job:
    """One schedulable unit. `index` is its row position in the source table."""
    index: int
    job_id: str
    tier: str
    arrival_time: float
    duration: float  # NaN for Gold
    cpu_demand: float
    mem_demand: float
    continuous: bool

    # filled in as the episode runs
    scheduled_time: Optional[float] = None
    machine: Optional[int] = None
    release_time: Optional[float] = None  # None for Gold: never releases
    # The simulated clock AT THE MOMENT the resources were actually handed back, stamped
    # inside _release_completions. Distinct from release_time (which is the *scheduled*
    # release): comparing the two is how the tests verify releases happen on time without
    # having to infer it from step-boundary snapshots, which lag by one clock jump.
    released_at: Optional[float] = None

    @property
    def is_gold(self) -> bool:
        return self.continuous

    def scheduling_delay(self, now: float) -> float:
        """Delay so far (if waiting) or final delay (if scheduled)."""
        end = self.scheduled_time if self.scheduled_time is not None else now
        return max(0.0, end - self.arrival_time)


# ----------------------------------------------------------------------------------
# REWARD (swappable - the RL stage will replace this)
# ----------------------------------------------------------------------------------

def default_reward(env: "ClusterSchedulingEnv", info: dict) -> float:
    """
    Core reward: PER-PLACEMENT delay attribution, plus utilization and a no-op penalty.

        reward = mean(cpu_util, mem_util)
                 - w_tier * (scheduled_time - arrival_time) / n_jobs_in_tier
                 - NOOP_PENALTY                               [only when nothing is placed]

    with w_tier = GOLD_PRIORITY_WEIGHT (10.0) or BRONZE_PRIORITY_WEIGHT (1.0). Dividing by
    the tier's job count makes the episode sum equal -w_tier * mean_delay_of_that_tier, so
    the ratio expresses average-delay-vs-average-delay rather than seconds-vs-seconds. See
    the constants above for the measured reason that distinction matters.

    WHY PER-PLACEMENT RATHER THAN QUEUE-INTEGRATED. The previous reward charged
    `dt * waiting_gold` every step - the delay accrued across the whole queue. That is a
    faithful description of the metric but a hopeless learning signal on this workload,
    and the first training run demonstrated why. Total Gold delay is about 1,142,000
    job-seconds, of which the 20 late-arriving Gold jobs - the ONLY jobs where a
    priority-aware policy can differ from FCFS - contribute roughly 8. The entire decision
    worth learning was about 0.001% of the return, while episode returns swung between
    -959 and +13,094. The agent optimised what it could see (utilization landed within
    0.008 of the baselines) and stayed blind to the rest, finishing at 806.8 s mean Gold
    delay and 772.5 s on the late cohort, against Static Priority's 745.6 and 0.4.

    Attributing the delay to the individual placement fixes the credit assignment: the
    reward for scheduling a particular job now depends on how long THAT job waited, so
    picking a freshly-arrived Gold job over a long-queued Bronze one is a locally visible
    win rather than a rounding error inside a million-second integral.

    WHY NOT gold_sla_violation_rate? Saturated - 98-99% across every policy and every tick
    value, varying by under 0.013 between the best and worst scheduler (see
    tick_sweep_results.csv). 1,512 of 1,532 Gold jobs arrive simultaneously at t=0, so with
    a 5 s threshold almost all breach regardless of queue order. It is reported in
    episode_metrics() as a descriptive statistic, never used as a training signal.

    Swap this out by passing any callable(env, info) -> float as `reward_fn`.
    """
    cpu_util, mem_util = env.utilization()
    util = UTIL_WEIGHT * 0.5 * (cpu_util + mem_util)

    # Continuous holding cost: every queued job ages by dt this step, priced with the same
    # per-tier weights and count normalisation as the placement term.
    dt = info.get("dt", 0.0)
    queue_term = 0.0
    if dt > 0.0:
        queue_term = -QUEUE_HOLD_FRACTION * dt * (
            GOLD_PRIORITY_WEIGHT * info.get("n_waiting_gold", 0) / max(env.n_gold_total, 1)
            + BRONZE_PRIORITY_WEIGHT * info.get("n_waiting_bronze", 0)
            / max(env.n_bronze_total, 1))
    gold_term = 0.0
    bronze_term = 0.0
    noop_term = 0.0

    # PLACEMENT CREDIT (positive), not a penalty. This is potential-based shaping with
    # Phi(s) = -sum over queued jobs of w_tier * wait_j / n_tier: placing a job removes its
    # accumulated wait from the queue, so the shaping reward is +w_tier * wait_j / n_tier.
    # It telescopes to ~0 across an episode, so it provably cannot change which policy is
    # optimal - it only supplies immediate per-decision credit for clearing a long-waited,
    # high-priority job.
    #
    # This term was previously applied with the OPPOSITE sign, as a penalty, alongside the
    # flow cost - which both double-counted the delay and inverted the incentive. Placing a
    # Bronze job that had waited 5,000 s cost -0.16 while a no-op cost -0.047, so deferring
    # was ~3x cheaper and the agent learned exactly that: measured greedy preference for the
    # no-op rose 7.2% -> 30.6% over five episodes, with Q(no-op) - max Q(job) climbing
    # -0.296 -> -0.014, i.e. heading for the point where doing nothing beats every
    # placement on offer. See diagnose_policy.py.
    #
    # Jobs never placed accrue flow cost and are never credited, so leaving work queued is
    # strictly worse than clearing it - which is the property the penalty version lacked.
    if info.get("scheduled", False):
        wait = info.get("placed_wait", 0.0)
        if info.get("placed_is_gold"):
            gold_term = +GOLD_PRIORITY_WEIGHT * wait / max(env.n_gold_total, 1)
        else:
            bronze_term = +BRONZE_PRIORITY_WEIGHT * wait / max(env.n_bronze_total, 1)
    else:
        noop_term = -NOOP_PENALTY

    # Expose the breakdown so training can report Gold vs Bronze contribution to the
    # return - the balance is easy to get wrong and impossible to see from the total.
    info["reward_components"] = {"util": util, "gold": gold_term, "bronze": bronze_term,
                                 "queue": queue_term, "noop": noop_term}

    return float(util + gold_term + bronze_term + queue_term + noop_term)


def freshness_bonus_reward(env: "ClusterSchedulingEnv", info: dict) -> float:
    """
    default_reward plus the reverted Gold freshness bonus. Kept so the negative result above
    can be reproduced; pass as `reward_fn` to re-run that experiment.
    """
    reward = default_reward(env, info)
    if info.get("scheduled", False) and info.get("placed_is_gold"):
        bonus = GOLD_FRESHNESS_BONUS * float(
            np.exp(-info.get("placed_wait", 0.0) / GOLD_FRESHNESS_TAU))
        info["reward_components"]["gold"] += bonus
        reward += bonus
    return float(reward)


def queue_delay_reward(env: "ClusterSchedulingEnv", info: dict) -> float:
    """
    The queue-integrated reward used in the first training run, kept for comparison:
    utilization minus GOLD_DELAY_WEIGHT * (dt * waiting_gold) / total_gold. See
    default_reward for the measured reason it was replaced.
    """
    cpu_util, mem_util = env.utilization()
    reward = 0.5 * (cpu_util + mem_util)
    gold_delay_accrued = info.get("dt", 0.0) * info.get("n_waiting_gold", 0)
    reward -= GOLD_DELAY_WEIGHT * gold_delay_accrued / max(env.n_gold_total, 1)
    if not info.get("scheduled", False):
        reward -= NOOP_PENALTY
    return float(reward)


def sla_penalty_reward(env: "ClusterSchedulingEnv", info: dict) -> float:
    """
    The original placeholder reward, kept for comparison: utilization minus a fixed penalty
    per new Gold SLA violation. Retained so the saturation argument in default_reward can
    be demonstrated rather than just asserted - train against this and the learning signal
    is nearly flat.
    """
    cpu_util, mem_util = env.utilization()
    reward = 0.5 * (cpu_util + mem_util)
    reward -= SLA_VIOLATION_PENALTY * info.get("new_sla_violations", 0)
    if not info.get("scheduled", False):
        reward -= NOOP_PENALTY
    return float(reward)


# ----------------------------------------------------------------------------------
# ENVIRONMENT
# ----------------------------------------------------------------------------------

class ClusterSchedulingEnv(gym.Env):
    """
    Gymnasium environment replaying jobs_clean.csv onto N_MACHINES machines.

    Observation (Box, float32), length 6 + 4*K:
        [0] cluster CPU utilization        in [0, 1]
        [1] cluster memory utilization     in [0, 1]
        [2] waiting Gold count             normalised by total jobs
        [3] waiting Bronze count           normalised by total jobs
        [4] mean Gold waiting time         log1p(seconds / SLA threshold)
        [5] mean Bronze waiting time       log1p(seconds / SLA threshold)
        then, for each of the K visible slots:
            tier flag (1.0 Gold / 0.0 Bronze), cpu_demand (norm), mem_demand,
            waiting time (log1p-compressed). All zero for padded slots.
    Waiting times are log-compressed rather than divided by the SLA threshold; see
    _scaled_wait for why the raw ratio breaks training.

    Action space: Discrete(K + 1). Actions 0..K-1 pick the corresponding visible job;
    action K is an explicit no-op that advances the clock. Invalid actions (empty slots)
    are reported through `info["action_mask"]`.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 jobs: Optional[pd.DataFrame] = None,
                 machines: Optional[pd.DataFrame] = None,
                 k_visible: int = K_VISIBLE,
                 sla_threshold: float = SLA_THRESHOLD_SECONDS,
                 reward_fn: Optional[Callable[["ClusterSchedulingEnv", dict], float]] = None,
                 window_mode: str = "per_tier",
                 scheduling_tick: float = SCHEDULING_TICK_SECONDS,
                 episode_buffer: float = EPISODE_BUFFER_SECONDS,
                 max_steps: Optional[int] = None,
                 render_mode: Optional[str] = None):
        super().__init__()

        self.jobs_df = jobs if jobs is not None else pd.read_csv(JOBS_FILE)
        if machines is None:
            machines = load_machine_capacities(machine_ids=default_machine_subset())
        self.machines_df = machines.reset_index(drop=True)

        self.k = int(k_visible)
        self.sla_threshold = float(sla_threshold)
        self.reward_fn = reward_fn or default_reward
        self.scheduling_tick = float(scheduling_tick)
        self.episode_buffer = float(episode_buffer)
        self.render_mode = render_mode

        # DESIGN - VISIBLE WINDOW MODE. With this trace all 1512 Gold jobs arrive at t=0
        # and Gold never releases, so a strict FIFO window fills permanently with Gold that
        # no longer fits anywhere: head-of-line blocking that starves Bronze completely and
        # makes every baseline collapse to the same degenerate numbers.
        #   "per_tier" (default) - half the slots to the oldest Gold, half to the oldest
        #                          Bronze, so both tiers stay visible and tier-aware
        #                          policies have a real choice to make.
        #   "fifo"               - strict arrival order. Faithful to a single queue, but
        #                          degenerate on this dataset. Kept for comparison.
        if window_mode not in ("per_tier", "fifo"):
            raise ValueError(f"unknown window_mode {window_mode!r}")
        self.window_mode = window_mode

        self.cpu_capacity = self.machines_df["capacity_cpu"].to_numpy(dtype=np.float64)
        self.mem_capacity = self.machines_df["capacity_mem"].to_numpy(dtype=np.float64)
        self.n_machines = len(self.machines_df)

        self.total_cpu = float(self.cpu_capacity.sum())
        self.total_mem = float(self.mem_capacity.sum())

        self.n_jobs = len(self.jobs_df)
        # Used to normalise the reward's Gold delay term into "mean seconds per Gold job".
        self.n_gold_total = int(self.jobs_df["continuous"].astype(bool).sum())
        self.n_bronze_total = self.n_jobs - self.n_gold_total
        self.horizon = float(self.jobs_df["arrival_time"].max()) + self.episode_buffer

        # Step bound: every step either places a job (at most n_jobs of those) or advances
        # the clock past an event. Events = arrivals + Bronze completions <= 2 * n_jobs.
        # 4 * n_jobs + 100 leaves generous slack while still catching a genuine hang.
        self.max_steps = int(max_steps if max_steps is not None else 4 * self.n_jobs + 100)

        obs_len = 6 + 4 * self.k
        self.observation_space = spaces.Box(low=0.0, high=np.inf,
                                            shape=(obs_len,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.k + 1)

        self._max_cpu_demand = max(float(self.jobs_df["cpu_demand"].max()), 1e-9)

        self.reset()

    # ---------------------------------------------------------------- lifecycle

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.jobs: list[Job] = [
            Job(index=i,
                job_id=row.job_id,
                tier=row.tier,
                arrival_time=float(row.arrival_time),
                duration=float(row.duration) if pd.notna(row.duration) else float("nan"),
                cpu_demand=float(row.cpu_demand),
                mem_demand=float(row.mem_demand),
                continuous=bool(row.continuous))
            for i, row in enumerate(self.jobs_df.itertuples(index=False))
        ]
        # Arrival order, so "not yet arrived" is a simple pointer walk.
        self._arrival_order = sorted(range(self.n_jobs),
                                     key=lambda i: (self.jobs[i].arrival_time, i))
        self._next_arrival_ptr = 0

        self.now = 0.0
        self.steps = 0

        self.alloc_cpu = np.zeros(self.n_machines, dtype=np.float64)
        self.alloc_mem = np.zeros(self.n_machines, dtype=np.float64)

        self.waiting: list[int] = []        # indices of arrived, unscheduled jobs
        self.running: set[int] = set()      # indices of scheduled, unreleased jobs
        # Min-heap of (release_time, index) for Bronze only. Gold never releases, so it
        # never enters the heap - which is the whole point: the old implementation
        # rescanned all ~1,500 permanently-running Gold jobs on every release check.
        self._release_heap: list[tuple[float, int]] = []
        self.scheduled: list[int] = []      # indices ever scheduled (running or completed)
        self.completed: list[int] = []      # Bronze jobs whose resources were released

        self._sla_violated: set[int] = set()  # Gold jobs already counted as violations

        # PERFORMANCE bookkeeping. All three of these replace a per-step O(queue) or
        # O(scheduled) scan with O(1) amortised work. They are pure accounting - the
        # simulation's behaviour is identical, which the correctness suite verifies.
        #
        # 1. Per-tier waiting queues in arrival order (admission appends in arrival order,
        #    so they stay sorted for free) with a head pointer, so the visible window is a
        #    slice instead of a full sort of the queue on every call.
        self._waiting_gold: list[int] = []
        self._waiting_bronze: list[int] = []
        self._head_gold = 0
        self._head_bronze = 0
        # 2. Running sums of arrival times per tier, so mean waiting time is
        #    now - mean(arrival) instead of a scan over every waiting job.
        self._wait_arrival_sum = {True: 0.0, False: 0.0}
        self._wait_count = {True: 0, False: 0}
        # 3. Gold jobs awaiting an SLA verdict, in arrival order. Because the clock only
        #    moves forward, once the front of this queue has not yet breached, nothing
        #    behind it has either - so the scan stops there.
        self._gold_pending: deque[int] = deque()
        self._pending_violations = 0

        # PERFORMANCE - per-step caches for the visible window, observation and action
        # mask. All three are pure functions of (clock, queue contents), so a version
        # counter bumped on every mutation is enough to keep them exact. Without this a
        # single step rebuilt the window three times (step, action_mask, observation) and
        # a policy that reads the observation rebuilt it twice more - greedy DQN evaluation
        # took ~8.5 min per episode against ~21 s for a baseline.
        self._cache_version = 0
        self._vis_cache: tuple[int, list[int] | None] = (-1, None)
        self._obs_cache: tuple[int, np.ndarray | None] = (-1, None)
        self._mask_cache: tuple[int, np.ndarray | None] = (-1, None)

        self._admit_arrivals()
        return self._observation(), self._info()

    # ---------------------------------------------------------------- clock / events

    def _invalidate(self) -> None:
        """Mark the per-step caches stale. Called on any clock or queue mutation."""
        self._cache_version += 1

    def _admit_arrivals(self) -> int:
        """Move every job with arrival_time <= now into the waiting list."""
        admitted = 0
        while self._next_arrival_ptr < self.n_jobs:
            idx = self._arrival_order[self._next_arrival_ptr]
            job = self.jobs[idx]
            if job.arrival_time > self.now:
                break
            self.waiting.append(idx)
            if job.is_gold:
                self._waiting_gold.append(idx)
                self._gold_pending.append(idx)
            else:
                self._waiting_bronze.append(idx)
            self._wait_arrival_sum[job.is_gold] += job.arrival_time
            self._wait_count[job.is_gold] += 1
            self._next_arrival_ptr += 1
            admitted += 1
        if admitted:
            self._invalidate()
        return admitted

    def _release_completions(self) -> int:
        """
        Free resources for every Bronze job whose release_time has been reached.

        Driven by a min-heap keyed on release_time, so the cost is O(log n) per actual
        release instead of O(running) per call. Gold jobs are never pushed onto the heap -
        they have release_time None and never release, which is the continuous-occupation
        model.
        """
        released = 0
        heap = self._release_heap
        while heap and heap[0][0] <= self.now:
            _, idx = heapq.heappop(heap)
            job = self.jobs[idx]
            job.released_at = self.now  # stamped at the instant of the release itself
            m = job.machine
            # Guard against float drift accumulating into a negative allocation.
            self.alloc_cpu[m] = max(0.0, self.alloc_cpu[m] - job.cpu_demand)
            self.alloc_mem[m] = max(0.0, self.alloc_mem[m] - job.mem_demand)
            self.running.discard(idx)
            self.completed.append(idx)
            released += 1
        if released:
            self._invalidate()
        return released

    def _next_event_time(self) -> Optional[float]:
        """
        Earliest arrival or Bronze release STRICTLY AFTER the current clock, or None if no
        future events remain.

        The strictness matters: 13 Bronze jobs in this trace have duration 0, so their
        release_time equals their scheduled_time. Without the `> self.now` filter such a
        job reports an event at the current instant, the clock cannot move to it, and the
        episode would fall through to the horizon while placeable work was still queued.
        Anything already due at `now` is settled by the caller before this is consulted.
        """
        times = []
        if self._next_arrival_ptr < self.n_jobs:
            nxt_arrival = self.jobs[self._arrival_order[self._next_arrival_ptr]].arrival_time
            if nxt_arrival > self.now:
                times.append(nxt_arrival)
        if self._release_heap and self._release_heap[0][0] > self.now:
            times.append(self._release_heap[0][0])
        return min(times) if times else None

    def _advance_clock(self) -> None:
        """
        Settle everything due at the current instant, then jump to the next future event.
        If no future events remain, jump to the horizon so the episode terminates instead
        of spinning on a permanently unschedulable backlog.
        """
        # Settle first: a duration-0 Bronze job placed this instant is already complete.
        self._release_completions()
        self._admit_arrivals()

        nxt = self._next_event_time()
        new_now = min(nxt, self.horizon) if nxt is not None else self.horizon
        if new_now != self.now:
            self.now = new_now
            self._invalidate()

        self._release_completions()
        self._admit_arrivals()

    # ---------------------------------------------------------------- placement

    def _first_fit(self, job: Job) -> Optional[int]:
        """
        DESIGN - FIRST-FIT machine selection. Machines are scanned in fixed index order and
        the first one with room takes the job. Chosen deliberately over best-fit/random:
        it is O(n_machines), deterministic (so tests and baseline comparisons are
        reproducible), and it keeps the *machine* choice out of the learning problem. The
        agent's decision is WHICH JOB to schedule; packing quality is held constant across
        every policy so comparisons isolate the priority decision.
        """
        for m in range(self.n_machines):
            if (self.alloc_cpu[m] + job.cpu_demand <= self.cpu_capacity[m] + 1e-9 and
                    self.alloc_mem[m] + job.mem_demand <= self.mem_capacity[m] + 1e-9):
                return m
        return None

    def can_place(self, job: Job) -> bool:
        """Public: does any machine currently have room for this job?"""
        return self._first_fit(job) is not None

    def _place(self, idx: int, machine: int) -> None:
        job = self.jobs[idx]
        job.scheduled_time = self.now
        job.machine = machine
        # Gold (continuous) never releases; Bronze releases at exactly scheduled + duration.
        job.release_time = None if job.continuous else self.now + job.duration

        self.alloc_cpu[machine] += job.cpu_demand
        self.alloc_mem[machine] += job.mem_demand

        self.waiting.remove(idx)
        self.running.add(idx)
        self.scheduled.append(idx)
        if job.release_time is not None:
            heapq.heappush(self._release_heap, (job.release_time, idx))
        self._invalidate()

        # Keep the O(1) bookkeeping in step with the queue. The per-tier lists use lazy
        # deletion (a placed job is identified by scheduled_time being set), so nothing is
        # removed from them here.
        self._wait_arrival_sum[job.is_gold] -= job.arrival_time
        self._wait_count[job.is_gold] -= 1

        # A Gold job's scheduling delay is final the moment it is placed, so assess its SLA
        # here rather than waiting for it to surface at the front of _gold_pending.
        if job.is_gold and idx not in self._sla_violated:
            if job.scheduled_time - job.arrival_time > self.sla_threshold:
                self._sla_violated.add(idx)
                self._pending_violations += 1

    # ---------------------------------------------------------------- observation

    def visible_jobs(self) -> list[int]:
        """
        The K job indices the scheduler can act on this step. Ordering is deterministic:
        oldest arrival first, ties broken by job index.
        """
        if self._vis_cache[0] == self._cache_version:
            return self._vis_cache[1]

        if not self.waiting:
            self._vis_cache = (self._cache_version, [])
            return []

        # Both per-tier queues are already in arrival order, so the oldest N are just the
        # first N still-unplaced entries. The head pointer skips the placed prefix.
        gold = self._front(self._waiting_gold, "_head_gold", self.k)
        bronze = self._front(self._waiting_bronze, "_head_bronze", self.k)

        if self.window_mode == "fifo":
            merged = gold + bronze
            merged.sort(key=lambda i: (self.jobs[i].arrival_time, i))
            self._vis_cache = (self._cache_version, merged[:self.k])
            return self._vis_cache[1]

        # per_tier: split the slots between tiers, then give any unused half to the other
        # tier so the window is never needlessly empty.
        half = self.k // 2
        take_gold = min(len(gold), max(half, self.k - len(bronze)))
        take_bronze = min(len(bronze), self.k - take_gold)
        self._vis_cache = (self._cache_version, gold[:take_gold] + bronze[:take_bronze])
        return self._vis_cache[1]

    def _front(self, queue: list[int], head_attr: str, k: int) -> list[int]:
        """
        First `k` still-waiting entries of an arrival-ordered queue, using lazy deletion:
        placed jobs are recognised by scheduled_time and skipped, and the head pointer is
        advanced past any contiguous placed prefix so the scan does not lengthen over time.
        """
        head = getattr(self, head_attr)
        while head < len(queue) and self.jobs[queue[head]].scheduled_time is not None:
            head += 1
        setattr(self, head_attr, head)

        out: list[int] = []
        i = head
        n = len(queue)
        while i < n and len(out) < k:
            idx = queue[i]
            if self.jobs[idx].scheduled_time is None:
                out.append(idx)
            i += 1

        # Holes left by lazy deletion accumulate whenever a job is placed from the middle
        # of the window rather than its front. Once the scan has to walk far past the head
        # to find k live entries, rebuild the queue and reset the pointer - amortised O(n),
        # and it keeps the common case at O(k).
        if i - head > 8 * k:
            live = [j for j in queue[head:] if self.jobs[j].scheduled_time is None]
            queue[:] = live
            setattr(self, head_attr, 0)
        return out

    def action_mask(self) -> np.ndarray:
        """Boolean mask over Discrete(K+1); the trailing no-op is always legal."""
        if self._mask_cache[0] == self._cache_version:
            return self._mask_cache[1]
        mask = np.zeros(self.k + 1, dtype=bool)
        mask[:len(self.visible_jobs())] = True
        mask[self.k] = True
        self._mask_cache = (self._cache_version, mask)
        return mask

    def utilization(self) -> tuple[float, float]:
        return (float(self.alloc_cpu.sum() / self.total_cpu),
                float(self.alloc_mem.sum() / self.total_mem))

    def _mean_wait(self, gold: bool) -> float:
        """
        Mean waiting time of the queued jobs of one tier.

        Every waiting job has arrival_time <= now, so mean(now - arrival) is exactly
        now - mean(arrival) - computed from running sums instead of a scan.
        """
        count = self._wait_count[gold]
        if count <= 0:
            return 0.0
        return max(0.0, self.now - self._wait_arrival_sum[gold] / count)

    def _scaled_wait(self, seconds: float) -> float:
        """
        Compress a waiting time into a network-friendly range.

        Dividing raw seconds by the 5 s SLA threshold produced observation components up to
        1,035 while every other feature sat in [0, 1] - three orders of magnitude apart.
        Fed to the DQN that diverged immediately (Q-values reached +781 within 3,000 steps
        on rewards bounded below by about -20). log1p keeps the ordering and the relative
        differences that matter while holding the feature around O(1-7).
        """
        return float(np.log1p(max(0.0, seconds) / self.sla_threshold))

    def _observation(self) -> np.ndarray:
        if self._obs_cache[0] == self._cache_version:
            return self._obs_cache[1]
        cpu_util, mem_util = self.utilization()
        n_gold = self._wait_count[True]
        n_bronze = self._wait_count[False]

        obs = [cpu_util, mem_util,
               n_gold / self.n_jobs, n_bronze / self.n_jobs,
               self._scaled_wait(self._mean_wait(True)),
               self._scaled_wait(self._mean_wait(False))]

        visible = self.visible_jobs()
        for slot in range(self.k):
            if slot < len(visible):
                job = self.jobs[visible[slot]]
                obs.extend([1.0 if job.is_gold else 0.0,
                            job.cpu_demand / self._max_cpu_demand,
                            job.mem_demand,
                            self._scaled_wait(job.scheduling_delay(self.now))])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0])  # padding for empty slots

        arr = np.asarray(obs, dtype=np.float32)
        self._obs_cache = (self._cache_version, arr)
        return arr

    # ---------------------------------------------------------------- SLA accounting

    def _count_new_sla_violations(self) -> int:
        """
        A Gold job violates its SLA once its scheduling delay passes the threshold. Counted
        once per job (tracked in _sla_violated) so a long wait is not billed every step.
        """
        new = self._pending_violations  # breaches detected at placement time in _place
        self._pending_violations = 0

        # _gold_pending is in arrival order, so the first job that has not yet breached
        # bounds every job behind it - the scan stops there instead of walking the queue.
        while self._gold_pending:
            idx = self._gold_pending[0]
            job = self.jobs[idx]
            if job.scheduled_time is not None:
                self._gold_pending.popleft()  # verdict already returned in _place
                continue
            if self.now - job.arrival_time > self.sla_threshold:
                self._gold_pending.popleft()
                if idx not in self._sla_violated:
                    self._sla_violated.add(idx)
                    new += 1
                continue
            break
        return new

    # ---------------------------------------------------------------- gym API

    def step(self, action: int):
        assert self.action_space.contains(int(action)), f"invalid action {action}"
        action = int(action)
        self.steps += 1

        # Clock position before the step, so the reward can charge the Gold delay actually
        # accrued (dt * queued Gold) rather than a proxy.
        t_before = self.now
        n_waiting_gold_before = self._wait_count[True]
        n_waiting_bronze_before = self._wait_count[False]

        visible = self.visible_jobs()
        scheduled_ok = False
        placed_idx = None

        if action < self.k and action < len(visible):
            idx = visible[action]
            machine = self._first_fit(self.jobs[idx])
            if machine is not None:
                self._place(idx, machine)
                scheduled_ok = True
                placed_idx = idx
                if self.scheduling_tick > 0:
                    # Settle releases due at the placement instant BEFORE the clock moves.
                    # Without this, a zero-duration Bronze job placed at T (release_time
                    # also T) is not noticed until the clock has already reached
                    # T + tick, so it holds its resources one full tick too long -
                    # exactly what correctness check 3 caught at tick=1.0.
                    self._release_completions()
                    self.now += self.scheduling_tick
                    self._invalidate()
                    self._release_completions()
                    self._admit_arrivals()
            # else: no machine has room -> no-op. The job stays in self.waiting and is
            # NOT marked scheduled. Penalised via the reward, nothing else changes.
        else:
            # Explicit no-op, or an action pointing at an empty slot: nothing more can be
            # done at this instant, so let the clock move to the next event.
            self._advance_clock()

        # A failed placement would otherwise leave the clock frozen and the episode could
        # spin. Advance on failure too, so progress is always made.
        if action < self.k and not scheduled_ok:
            self._advance_clock()

        new_violations = self._count_new_sla_violations()

        info = self._info()
        info.update({"scheduled": scheduled_ok,
                     "placed_job": self.jobs[placed_idx].job_id if placed_idx is not None else None,
                     "new_sla_violations": new_violations,
                     "dt": self.now - t_before,
                     "n_waiting_gold": n_waiting_gold_before,
                     "n_waiting_bronze": n_waiting_bronze_before,
                     # Wait accumulated by the job placed this step, for the per-placement
                     # delay term in default_reward.
                     "placed_wait": (self.jobs[placed_idx].scheduled_time
                                     - self.jobs[placed_idx].arrival_time
                                     if placed_idx is not None else 0.0),
                     "placed_is_gold": (self.jobs[placed_idx].is_gold
                                        if placed_idx is not None else False)})

        reward = self.reward_fn(self, info)

        # TIME LIMITS ARE TRUNCATION, NOT TERMINATION. Reaching the horizon (or the step
        # cap) means the simulation window ended, not that the world stopped - there is
        # still a queue with future value, so the agent must BOOTSTRAP through it rather
        # than value it at zero.
        #
        # Returning horizon-reached as `terminated` made the horizon a free absorbing
        # state. Once the utilization weight was cut and nearly every reward went negative,
        # the highest-value action became "reach the exit as fast as possible" - and since
        # a no-op calls _advance_clock(), which jumps straight to the next event, stalling
        # was the quickest route there. Steps per episode duly fell 44,158 -> 37,360 over 20
        # episodes while Bronze waiting nearly tripled. The agent was not avoiding
        # placements; it was sprinting for the exit, and skipping placements is how.
        terminated = self._episode_completed()
        truncated = (not terminated) and (self.now >= self.horizon
                                          or self.steps >= self.max_steps)
        return self._observation(), float(reward), bool(terminated), bool(truncated), info

    def _episode_completed(self) -> bool:
        """
        GENUINE terminal state: every job has arrived and none is still waiting. There is
        no future left to value, so bootstrapping zero here is correct.
        """
        return self._next_arrival_ptr >= self.n_jobs and not self.waiting

    def _episode_done(self) -> bool:
        """Episode over for any reason - genuine completion or the time limit."""
        return self._episode_completed() or self.now >= self.horizon

    def _info(self) -> dict:
        cpu_util, mem_util = self.utilization()
        return {
            "time": self.now,
            "steps": self.steps,
            "cpu_util": cpu_util,
            "mem_util": mem_util,
            "n_waiting": len(self.waiting),
            "n_running": len(self.running),
            "n_scheduled": len(self.scheduled),
            "n_not_arrived": self.n_jobs - self._next_arrival_ptr,
            "action_mask": self.action_mask(),
        }

    # ---------------------------------------------------------------- reporting

    def episode_metrics(self) -> dict:
        """Summary metrics for the finished (or in-progress) episode."""
        gold = [j for j in self.jobs if j.is_gold]
        bronze = [j for j in self.jobs if not j.is_gold]

        gold_sched = [j for j in gold if j.scheduled_time is not None]
        bronze_sched = [j for j in bronze if j.scheduled_time is not None]

        # An unscheduled Gold job is an SLA violation too: it waited the whole episode.
        gold_violations = sum(1 for j in gold
                              if j.scheduled_time is None
                              or j.scheduling_delay(self.now) > self.sla_threshold)

        def mean(xs):
            return float(np.mean(xs)) if xs else 0.0

        # METRIC DESIGN - unscheduled Gold must not be free.
        # Averaging delay over scheduled Gold only creates a perverse incentive: a policy
        # that never places a slow job improves its own score by dropping it from the
        # denominator. That is not hypothetical - in the tick sweep RoundRobin posted a
        # LOWER average Gold delay than FCFS at tick=10 while scheduling half as many Gold
        # jobs. So the headline number charges every unscheduled Gold job its censored wait
        # (episode end - arrival), which is a lower bound on the delay it would have had.
        # The scheduled-only figure is kept alongside for comparison, together with an
        # explicit count, so the two can never be confused.
        gold_unscheduled = [j for j in gold if j.scheduled_time is None]
        gold_delay_all = ([j.scheduling_delay(self.now) for j in gold_sched] +
                          [self.now - j.arrival_time for j in gold_unscheduled])

        cpu_util, mem_util = self.utilization()
        return {
            "gold_sla_violation_rate": gold_violations / len(gold) if gold else 0.0,
            "gold_avg_scheduling_delay": mean(gold_delay_all),
            "gold_avg_scheduling_delay_scheduled_only": mean(
                [j.scheduling_delay(self.now) for j in gold_sched]),
            "n_gold_unscheduled": len(gold_unscheduled),
            "bronze_avg_completion_time": mean([
                (j.release_time - j.arrival_time) for j in bronze_sched
                if j.release_time is not None]),
            "cpu_utilization": cpu_util,
            "mem_utilization": mem_util,
            "gold_avg_waiting_time": mean([j.scheduling_delay(self.now) for j in gold]),
            "bronze_avg_waiting_time": mean([j.scheduling_delay(self.now) for j in bronze]),
            "throughput": len(self.scheduled) / max(self.now, 1e-9),
            "n_gold_scheduled": len(gold_sched),
            "n_bronze_scheduled": len(bronze_sched),
            "n_scheduled": len(self.scheduled),
            "n_unscheduled": self.n_jobs - len(self.scheduled),
            "sim_time": self.now,
            "steps": self.steps,
        }

    def render(self):
        """Debug print of the current cluster state."""
        cpu_util, mem_util = self.utilization()
        n_gold_wait = sum(1 for i in self.waiting if self.jobs[i].is_gold)
        busy = int(np.sum((self.alloc_cpu > 0) | (self.alloc_mem > 0)))

        print(f"t={self.now:9.1f}s  step={self.steps:6d}  "
              f"cpu={cpu_util:6.1%}  mem={mem_util:6.1%}  machines_busy={busy}/{self.n_machines}")
        print(f"   waiting={len(self.waiting):5d} (gold {n_gold_wait}, "
              f"bronze {len(self.waiting) - n_gold_wait})  running={len(self.running):5d}  "
              f"scheduled={len(self.scheduled):5d}  not_arrived={self.n_jobs - self._next_arrival_ptr:5d}")

        visible = self.visible_jobs()
        if visible:
            print("   visible window:")
            for slot, idx in enumerate(visible):
                job = self.jobs[idx]
                fits = self.can_place(job)
                print(f"     [{slot}] {job.tier:6s} {job.job_id:22s} "
                      f"cpu={job.cpu_demand:5.2f} mem={job.mem_demand:.4f} "
                      f"wait={job.scheduling_delay(self.now):7.1f}s "
                      f"{'fits' if fits else 'NO ROOM'}")
        else:
            print("   visible window: (empty)")
