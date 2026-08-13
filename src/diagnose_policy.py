"""
Read-only diagnostic: what is the learned policy actually doing?

Five smoke tests told us steps-per-episode falls ~13-15% while TD error stays healthy, and
two proposed mechanisms (avoiding placements, racing for the episode exit) were both wrong.
This measures the behaviour directly instead of inferring it.

Per episode it records:

  ACTIONS TAKEN (what actually happened, blending exploration and exploitation)
    place_ok    - action pointed at a visible job and it was placed
    place_fail  - action pointed at a visible job that fits on no machine (a wasted step)
    noop        - the explicit no-op action (index K)
    empty_slot  - action pointed at a slot with no job in it

  GREEDY INTENT (what the network WOULD do, computed every step regardless of epsilon)
    This is the key measurement. The greedy choice is argmax over masked Q-values and does
    not depend on epsilon at all, so it is fully observable even when behaviour is 90%
    random. It is what the earlier smoke tests could not see.
      greedy_noop      - the network's preferred action is the no-op
      greedy_job_fits  - it prefers a job that can actually be placed
      greedy_job_stuck - it prefers a job that fits nowhere
    q_gap = Q[no-op] - max Q[job slots], averaged. Positive means the network values
    doing nothing above every available placement.

Training still runs normally (the network learns); nothing about the design is changed.

Run:  python3 diagnose_policy.py [n_episodes]
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from baselines import run_episode
from dqn_agent import (DQNAgent, REWARD_CLIP, REWARD_SCALE, TRAIN_EVERY, late_gold_delay)
from environment import CSV_DIR, ClusterSchedulingEnv, ensure_output_dirs


def greedy_rollout(agent: DQNAgent, seed: int = 0) -> dict:
    """
    One full episode with exploration switched off, to read the LEARNED policy's realized
    metrics. Runs on a fresh environment and leaves the training epsilon schedule alone -
    lowering epsilon for diagnostics would make the run non-comparable to real training.
    """
    was_greedy = agent.greedy
    agent.greedy = True
    try:
        env = ClusterSchedulingEnv()
        m = run_episode(env, agent, seed=seed)
        return {"g_gold_delay": m["gold_avg_scheduling_delay"],
                "g_bronze_wait": m["bronze_avg_waiting_time"],
                "g_late_gold": late_gold_delay(env),
                "g_steps": m["steps"],
                "g_unscheduled": m["n_unscheduled"]}
    finally:
        agent.greedy = was_greedy


def diagnose(n_episodes: int = 5, seed: int = 0) -> pd.DataFrame:
    env = ClusterSchedulingEnv()
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, seed=seed)
    rng = np.random.default_rng(seed + 999)  # our own draw, so greedy/random is observable
    k = env.k

    rows = []
    print(f"Environment : {env.n_jobs:,} jobs, {env.n_machines} machines, "
          f"tick={env.scheduling_tick}")
    print(f"Diagnostic  : {n_episodes} episodes, greedy intent sampled every step\n")

    header = (f"{'ep':>3s} {'steps':>7s} | {'place_ok':>9s} {'place_fail':>10s} "
              f"{'noop':>8s} | {'g_noop%':>8s} {'g_fits%':>8s} {'q_gap':>9s} | "
              f"{'GREEDY: gold':>13s} {'bronze':>10s} {'late_g':>8s} {'unsch':>7s} | "
              f"{'sec':>5s}")
    print(header)
    print("-" * len(header))

    for ep in range(1, n_episodes + 1):
        t0 = time.time()
        obs, info = env.reset(seed=seed + ep)
        mask = info["action_mask"].copy()

        c = dict(place_ok=0, place_fail=0, noop=0, empty=0,
                 g_noop=0, g_fits=0, g_stuck=0)
        q_gaps = []

        while True:
            # ---- greedy intent, measured every step and independent of epsilon --------
            q = agent.online.forward(obs[None, :].astype(np.float32))[0]
            q_masked = np.where(mask, q, -np.inf)
            greedy_action = int(np.argmax(q_masked))

            visible = env.visible_jobs()
            if greedy_action == k:
                c["g_noop"] += 1
            elif greedy_action < len(visible):
                if env.can_place(env.jobs[visible[greedy_action]]):
                    c["g_fits"] += 1
                else:
                    c["g_stuck"] += 1

            # Q(no-op) vs the best Q among slots holding a real job.
            job_slots = q_masked[:min(len(visible), k)]
            if job_slots.size and np.isfinite(job_slots).any():
                q_gaps.append(float(q[k] - np.max(job_slots)))

            # ---- behaviour: epsilon-greedy, with the coin flip drawn here ------------
            if rng.random() < agent.epsilon():
                action = int(rng.choice(np.flatnonzero(mask)))
            else:
                action = greedy_action

            # ---- classify what the action actually did -------------------------------
            if action == k:
                c["noop"] += 1
            elif action >= len(visible):
                c["empty"] += 1

            next_obs, reward, terminated, truncated, info = env.step(action)

            if action < k and action < len(visible):
                c["place_ok" if info["scheduled"] else "place_fail"] += 1

            next_mask = info["action_mask"].copy()
            shaped = float(np.clip(reward * REWARD_SCALE, -REWARD_CLIP, REWARD_CLIP))
            agent.buffer.add(obs, action, shaped, next_obs, float(terminated), next_mask)

            agent.total_steps += 1
            if agent.total_steps % TRAIN_EVERY == 0:
                if np.isfinite(agent.train_step()):
                    agent.soft_update()

            obs, mask = next_obs, next_mask
            if terminated or truncated:
                break

        roll = greedy_rollout(agent, seed=seed)
        m = env.episode_metrics()
        steps = m["steps"]
        pct = lambda v: 100.0 * v / max(steps, 1)  # noqa: E731
        row = dict(episode=ep, steps=steps, **c, **roll,
                   q_gap=float(np.mean(q_gaps)) if q_gaps else float("nan"),
                   epsilon=agent.epsilon(),
                   greedy_share=1.0 - agent.epsilon(),
                   gold_delay=m["gold_avg_scheduling_delay"],
                   bronze_wait=m["bronze_avg_waiting_time"],
                   late_gold=late_gold_delay(env))
        rows.append(row)

        print(f"{ep:3d} {steps:7,d} | {c['place_ok']:9,d} {c['place_fail']:10,d} "
              f"{c['noop']:8,d} | {pct(c['g_noop']):7.1f}% {pct(c['g_fits']):7.1f}% "
              f"{row['q_gap']:9.4f} | {roll['g_gold_delay']:13,.1f} "
              f"{roll['g_bronze_wait']:10,.1f} {roll['g_late_gold']:8,.1f} "
              f"{roll['g_unscheduled']:7,d} | {time.time() - t0:5.0f}", flush=True)

    return pd.DataFrame(rows)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    df = diagnose(n)
    ensure_output_dirs()
    df.to_csv(CSV_DIR / "policy_diagnostic.csv", index=False)

    first, last = df.iloc[0], df.iloc[-1]
    steps = df["steps"]

    print("\n" + "=" * 78)
    print("DIAGNOSIS")
    print("=" * 78)
    print(f"  steps/episode          {first['steps']:>10,.0f} -> {last['steps']:>10,.0f}")
    print(f"  successful placements  {first['place_ok']:>10,.0f} -> {last['place_ok']:>10,.0f}")
    print(f"  failed placements      {first['place_fail']:>10,.0f} -> {last['place_fail']:>10,.0f}")
    print(f"  explicit no-ops        {first['noop']:>10,.0f} -> {last['noop']:>10,.0f}")
    print()
    print("  GREEDY INTENT (share of steps where the network's argmax would...)")
    for label, col in (("do nothing (no-op)", "g_noop"),
                       ("place a job that fits", "g_fits"),
                       ("pick a job that fits nowhere", "g_stuck")):
        a = 100.0 * first[col] / max(first["steps"], 1)
        b = 100.0 * last[col] / max(last["steps"], 1)
        print(f"    {label:32s} {a:6.1f}% -> {b:6.1f}%")
    print()
    print(f"  Q(no-op) - max Q(job)  {first['q_gap']:>10.4f} -> {last['q_gap']:>10.4f}")
    print("    positive => the network prefers doing nothing to every placement on offer")
    print()
    print("  GREEDY ROLLOUT (exploration off - the learned policy's realized metrics)")
    print(f"    gold delay             {first['g_gold_delay']:>10,.1f} -> {last['g_gold_delay']:>10,.1f}"
          f"   (StaticPriority 745.6)")
    print(f"    bronze wait            {first['g_bronze_wait']:>10,.1f} -> {last['g_bronze_wait']:>10,.1f}"
          f"   (StaticPriority 1,018.9)")
    print(f"    late gold delay        {first['g_late_gold']:>10,.1f} -> {last['g_late_gold']:>10,.1f}"
          f"   (StaticPriority 0.4)")
    print(f"    jobs left unscheduled  {first['g_unscheduled']:>10,.0f} -> {last['g_unscheduled']:>10,.0f}")
    print()
    print(f"  greedy action share    {first['greedy_share']:>9.1%} -> {last['greedy_share']:>9.1%}")
    if last["greedy_share"] < 0.25:
        print("    NOTE: behaviour is still mostly random, so the ACTIONS TAKEN columns")
        print("    reflect exploration more than policy. The GREEDY INTENT columns and")
        print("    q_gap above are epsilon-independent and are the ones to read.")
    print("=" * 78)
    print(f"\nWritten to {CSV_DIR / 'policy_diagnostic.csv'}")


if __name__ == "__main__":
    main()
