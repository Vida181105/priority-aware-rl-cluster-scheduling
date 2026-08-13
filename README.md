# Priority-Aware RL Scheduling for Cloud Clusters

A priority-aware DQN scheduler for multi-tenant cloud clusters, trained and evaluated on the
Alibaba Cluster Trace v2017 (`trace_201708`). The trace is cleaned into a unified job stream of
two tiers — **Gold** (long-running online service containers, latency-sensitive, which hold
their resources for the rest of the episode) and **Bronze** (batch tasks, throughput-oriented,
which release resources on completion) — and replayed through a custom Gymnasium environment
modelling 150 real machines with capacities taken from the trace. A DQN agent decides which
queued job to place next, and is measured against four scheduling policies already used in
production systems: FCFS, Round Robin, Static Priority, and Resource Reservation. The
environment is backed by 8 automated correctness checks covering resource conservation, tier
release semantics, job conservation and episode termination, so that results rest on a
simulator proven correct rather than assumed correct.

## Structure

```
project/
├── data/                              raw trace + generated jobs_clean.csv
├── notebooks/
│   └── 01_data_preparation.ipynb      trace cleaning -> data/jobs_clean.csv
├── src/
│   ├── environment.py                 Gymnasium cluster simulator + reward functions
│   ├── baselines.py                   FCFS / RoundRobin / StaticPriority / ResourceReservation
│   ├── dqn_agent.py                   NumPy Double-DQN, training + evaluation
│   ├── diagnose_policy.py             read-only policy instrumentation
│   └── sweep_scheduling_tick.py       SCHEDULING_TICK_SECONDS sweep
├── tests/
│   └── test_environment.py            8 automated correctness checks
├── models/
│   └── dqn_weights.npz                shipped model (best greedy checkpoint)
├── results/
│   ├── csv/                           metrics, sweeps, checkpoints
│   ├── logs/                          raw run logs
│   └── plots/                         training curves
└── README.md
```

All scripts anchor their paths on `src/environment.py`'s location, so they can be run from the
project root or any other working directory.

## Running things, in order

Requires Python 3.11 with `numpy`, `pandas`, `gymnasium`, `matplotlib` (CPU only — no GPU, no
PyTorch).

**1. Prepare the data** — produces `data/jobs_clean.csv` (1,532 Gold + 40,649 Bronze jobs):

```bash
jupyter notebook notebooks/01_data_preparation.ipynb     # run all cells
```

**2. Verify the simulator** — all 8 checks must pass before any result is meaningful:

```bash
python3 tests/test_environment.py        # or: pytest tests/test_environment.py -v
```

**3. Evaluate the baselines** — runs all four policies over the full trace:

```bash
python3 src/baselines.py
```

**4. Train the DQN** — optional episode count (default 50); greedy evaluation checkpoints every
5 episodes, and the best checkpoint is what gets saved, not the final episode:

```bash
python3 src/dqn_agent.py            # 50 episodes, ~35 min
python3 src/dqn_agent.py 20         # shorter smoke run
```

Writes `models/dqn_weights.npz`, `results/csv/dqn_*.csv` and `results/plots/dqn_training.png`.

**Supporting analyses** (not needed for the main pipeline):

```bash
python3 src/sweep_scheduling_tick.py     # why SCHEDULING_TICK_SECONDS is locked at 1.0
python3 src/diagnose_policy.py 5         # what the learned policy actually does, step by step
```

## Current result

The shipped model (best greedy checkpoint) reaches **parity with the hand-tuned baselines** on
aggregate metrics — Gold scheduling delay 757.7 s vs Static Priority's 745.6 s, Bronze waiting
1,021.6 s vs 1,018.9 s, identical throughput and CPU utilisation, with all 42,181 jobs
scheduled. It does **not** beat them on late-arriving Gold jobs (79.4 s vs 0.4 s), the cohort
that most sharply distinguishes a priority-aware scheduler. That cohort is only 20 jobs out of
42,181, and checkpoint-to-checkpoint variance on it spans two orders of magnitude, so resolving
it would need a multi-seed study reporting mean ± standard deviation rather than single runs.

Design decisions and the reward-shaping failures found along the way are documented inline in
`src/environment.py` — including the memory-unit mismatch between the trace's two capacity
columns, why the SLA violation rate is unusable as a training signal, and the reward
formulations that were tried and reverted.
