"""
DQN agent for the priority-aware cluster scheduler.

Implemented in NumPy only (no torch/TF), matching the project's CPU-only
NumPy/Pandas/Gymnasium/Matplotlib stack. The network is small - 46 inputs, two hidden
layers of 128, 11 outputs - so a hand-written MLP with Adam is entirely adequate and keeps
the dependency surface at zero.

Architecture notes:
  * DOUBLE DQN: the online net selects the next action, the target net evaluates it. Plain
    DQN takes a max over target Q-values, which systematically overestimates and here fed
    straight back through the bootstrap - the first stable-reward run still diverged, mean
    |TD error| climbing 0.046 -> 5.02 over 20 episodes on rewards bounded in [-1, +0.1].
  * Soft (Polyak) target updates and a 3e-4 learning rate, for the same reason.
  * ACTION MASKING is applied both when acting and when bootstrapping. The environment's
    action space is Discrete(K+1) but fewer than K jobs are often visible; without masking
    the agent would learn Q-values for slots that do not exist and bootstrap off them.
  * Reward is the environment's `default_reward` (utilization - weighted Gold scheduling
    delay), scaled by REWARD_SCALE purely for numerical conditioning of the network. The
    scaling never touches reported metrics.

TRAINING SCALE - why the full environment. The obvious speed-up is to subsample Bronze,
but that was measured and it destroys the thing being learned: at stride 5 and 10 the
Bronze backlog drains, and FCFS and Static Priority collapse back to identical late-Gold
delays (0.40 s each). The contention IS the Bronze volume. So training runs on the real
42,181-job environment at ~43.6k steps per episode.

Run:  python3 dqn_agent.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baselines import (FCFSScheduler, ResourceReservationScheduler, RoundRobinScheduler,
                       StaticPriorityScheduler, run_episode)
from environment import (BRONZE_PRIORITY_WEIGHT, CSV_DIR, ClusterSchedulingEnv,
                         GOLD_PRIORITY_WEIGHT, MODEL_DIR, PLOT_DIR, ensure_output_dirs)

# ----------------------------------------------------------------------------------
# HYPERPARAMETERS
# ----------------------------------------------------------------------------------

HIDDEN = (128, 128)
# 0.95 rather than 0.99: at 0.99 the effective horizon is ~100 steps, but a placement
# decision here is largely myopic - take this job now or don't - so the long credit chain
# bought nothing and made the value function harder to fit.
GAMMA = 0.95
LEARNING_RATE = 3e-4
BATCH_SIZE = 64
# SIZED FROM THIS ENVIRONMENT. An episode runs ~43,600 steps, so:
#   buffer   200,000 transitions ~= 4.6 episodes (was 100,000 ~= 2.3, which overwrote the
#            late-episode states around t=37,000s before they could be learned from)
#   epsilon  decays over 4.5M steps ~= 103 episodes, about 70% of the run (was 150,000
#            steps, which floored epsilon at 0.05 by episode 4 of 25 - almost the entire
#            first run was spent exploiting a barely-trained network)
BUFFER_CAPACITY = 200_000
TRAIN_EVERY = 4              # gradient step every N environment steps
# Soft target updates (Polyak) applied on every gradient step, replacing a hard copy every
# 2,000 env steps. The hard copy moved the bootstrap target in periodic jumps, which fed the
# instability; tau=0.005 over ~10,900 updates per episode gives a ~200-update time constant.
TAU = 0.005
WARMUP_STEPS = 5_000
EPS_START, EPS_END = 1.0, 0.05
# RESCALED to the actual 50-episode budget: ~43,600 steps/episode x 50 = ~2.18M steps, so
# decaying over 1.5M floors epsilon around episode 34 and leaves ~16 episodes of genuine
# exploitation. The previous 4.5M value was sized for an abandoned 150-episode run and left
# the agent at epsilon 0.53 when training ended - it never had an exploitation phase at all.
EPS_DECAY_STEPS = 1_500_000
N_EPISODES = 50
EVAL_EVERY = 5           # greedy checkpoint cadence, in episodes
GRAD_CLIP = 10.0

# Rewards run to roughly [-10, +1] per step under the count-normalised reward (a Gold job
# placed after the full 1,511 s t=0 backlog costs -10 * 1511/1532 = -9.86). Scaling by 0.1
# puts the working range in about [-1, +0.1], then clipping guards the tail - the DQN
# paper's reward-clipping trick, applied after scaling so it almost never binds and the
# gradient between "Gold waited 100 s" and "Gold waited 1,000 s" is preserved.
#
# The previous run diverged with these left unclipped: mean |TD error| climbed 0.147 -> 2.55
# -> 108.8 over episodes 9-33, on returns near -250,000.
# Both are learning-only; every reported metric is computed from raw rewards.
# LOWERED 0.1 -> 0.02: under the flow-based reward the queue cost reaches ~-9.87/step
# during the t=0 Gold burst, which at 0.1 scaled to -0.99 and sat on the clip - flattening
# ~1,500 burst steps to a constant -1.0 with no gradient between actions. At 0.02 the burst
# lands near -0.2 and stays informative.
REWARD_SCALE = 0.02
REWARD_CLIP = 1.0

SEED = 0


# ----------------------------------------------------------------------------------
# NETWORK
# ----------------------------------------------------------------------------------

class MLP:
    """Two-hidden-layer ReLU network with He initialisation and an Adam optimiser."""

    def __init__(self, n_in: int, n_out: int, hidden=HIDDEN, lr=LEARNING_RATE, seed=0):
        rng = np.random.default_rng(seed)
        sizes = [n_in, *hidden, n_out]
        self.W = [rng.normal(0, np.sqrt(2.0 / sizes[i]), (sizes[i], sizes[i + 1]))
                  for i in range(len(sizes) - 1)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.lr = lr
        # Adam state
        self._mW = [np.zeros_like(w) for w in self.W]
        self._vW = [np.zeros_like(w) for w in self.W]
        self._mb = [np.zeros_like(b) for b in self.b]
        self._vb = [np.zeros_like(b) for b in self.b]
        self._t = 0

    def forward(self, x: np.ndarray, cache: bool = False):
        acts = [x]
        h = x
        for i in range(len(self.W) - 1):
            h = np.maximum(0.0, h @ self.W[i] + self.b[i])
            acts.append(h)
        out = h @ self.W[-1] + self.b[-1]
        return (out, acts) if cache else out

    def backward(self, acts, grad_out):
        """Backprop `grad_out` (dL/dout) and apply one Adam step."""
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)

        g = grad_out
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = acts[i].T @ g
            gb[i] = g.sum(axis=0)
            if i > 0:
                g = (g @ self.W[i].T) * (acts[i] > 0)

        self._adam(gW, gb)

    def _adam(self, gW, gb, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        for i in range(len(self.W)):
            for grad, param, m, v in ((gW[i], self.W[i], self._mW, self._vW),
                                      (gb[i], self.b[i], self._mb, self._vb)):
                norm = np.linalg.norm(grad)
                if norm > GRAD_CLIP:
                    grad = grad * (GRAD_CLIP / norm)
                m[i] = beta1 * m[i] + (1 - beta1) * grad
                v[i] = beta2 * v[i] + (1 - beta2) * (grad ** 2)
                mhat = m[i] / (1 - beta1 ** self._t)
                vhat = v[i] / (1 - beta2 ** self._t)
                param -= self.lr * mhat / (np.sqrt(vhat) + eps)

    def copy_from(self, other: "MLP"):
        self.W = [w.copy() for w in other.W]
        self.b = [b.copy() for b in other.b]


# ----------------------------------------------------------------------------------
# REPLAY BUFFER
# ----------------------------------------------------------------------------------

class ReplayBuffer:
    """Uniform replay. Stores the next-state action mask so bootstrapping can be masked."""

    def __init__(self, capacity: int, obs_dim: int, n_actions: int, seed: int = 0):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.next_mask = np.zeros((capacity, n_actions), dtype=bool)
        self.size = 0
        self._ptr = 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, next_obs, done, next_mask):
        i = self._ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = done
        self.next_mask[i] = next_mask
        self._ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (self.obs[idx], self.actions[idx], self.rewards[idx],
                self.next_obs[idx], self.dones[idx], self.next_mask[idx])


# ----------------------------------------------------------------------------------
# AGENT
# ----------------------------------------------------------------------------------

@dataclass
class TrainingLog:
    episode: list = field(default_factory=list)
    total_reward: list = field(default_factory=list)
    gold_delay: list = field(default_factory=list)
    late_gold_delay: list = field(default_factory=list)
    bronze_wait: list = field(default_factory=list)
    cpu_util: list = field(default_factory=list)
    epsilon: list = field(default_factory=list)
    r_gold: list = field(default_factory=list)
    r_bronze: list = field(default_factory=list)
    r_util: list = field(default_factory=list)
    r_noop: list = field(default_factory=list)
    r_queue: list = field(default_factory=list)
    loss: list = field(default_factory=list)
    q_mag: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    wall_time: list = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({k: v for k, v in self.__dict__.items()})


class DQNAgent:
    """Deep Q-Network with target network, uniform replay and action masking."""

    name = "DQN"

    def __init__(self, obs_dim: int, n_actions: int, seed: int = SEED):
        self.n_actions = n_actions
        self.online = MLP(obs_dim, n_actions, seed=seed)
        self.target = MLP(obs_dim, n_actions, seed=seed)
        self.target.copy_from(self.online)
        self.buffer = ReplayBuffer(BUFFER_CAPACITY, obs_dim, n_actions, seed=seed)
        self.rng = np.random.default_rng(seed)
        self.total_steps = 0
        self.last_q_abs = float("nan")
        self.greedy = False  # flipped on for evaluation

    # -- acting ---------------------------------------------------------------------

    def epsilon(self) -> float:
        if self.greedy:
            return 0.0
        frac = min(1.0, self.total_steps / EPS_DECAY_STEPS)
        return EPS_START + frac * (EPS_END - EPS_START)

    def act(self, obs: np.ndarray, mask: np.ndarray) -> int:
        legal = np.flatnonzero(mask)
        if not self.greedy and self.rng.random() < self.epsilon():
            return int(self.rng.choice(legal))
        q = self.online.forward(obs[None, :].astype(np.float32))[0]
        # Mask before the argmax so illegal slots can never be chosen.
        q = np.where(mask, q, -np.inf)
        return int(np.argmax(q))

    def __call__(self, env: ClusterSchedulingEnv) -> int:
        """Policy interface, so the agent drops straight into run_episode()."""
        return self.act(env._observation(), env.action_mask())

    # -- learning -------------------------------------------------------------------

    def train_step(self) -> float:
        if self.buffer.size < max(WARMUP_STEPS, BATCH_SIZE):
            return float("nan")

        obs, actions, rewards, next_obs, dones, next_mask = self.buffer.sample(BATCH_SIZE)

        # DOUBLE DQN bootstrap: the ONLINE net picks the next action, the TARGET net scores
        # it. Decoupling selection from evaluation removes the max-operator's optimistic
        # bias, which is what the single-net version was accumulating. Illegal next-actions
        # are masked out of the selection, never merely out of the evaluation.
        rows_b = np.arange(BATCH_SIZE)
        q_next_online = self.online.forward(next_obs)
        q_next_online = np.where(next_mask, q_next_online, -np.inf)
        next_actions = np.argmax(q_next_online, axis=1)
        q_next_target = self.target.forward(next_obs)
        best_next = q_next_target[rows_b, next_actions]
        targets = rewards + GAMMA * (1.0 - dones) * best_next

        q_all, acts = self.online.forward(obs, cache=True)
        rows = np.arange(BATCH_SIZE)
        q_taken = q_all[rows, actions]
        # Mean |Q| over the replay batch. Logged so a climbing TD error can be attributed:
        # if |Q| grows with it, the value function is diverging; if |Q| is flat, the TD
        # error is coming from a widening state distribution instead.
        self.last_q_abs = float(np.mean(np.abs(q_all)))

        # Huber (smooth L1) gradient: clip the error to keep outliers from dominating.
        err = q_taken - targets
        grad_taken = np.clip(err, -1.0, 1.0) / BATCH_SIZE

        grad_out = np.zeros_like(q_all)
        grad_out[rows, actions] = grad_taken
        self.online.backward(acts, grad_out)

        return float(np.mean(np.abs(err)))

    def soft_update(self, tau: float = TAU):
        """Polyak averaging: target <- tau * online + (1 - tau) * target."""
        for i in range(len(self.online.W)):
            self.target.W[i] += tau * (self.online.W[i] - self.target.W[i])
            self.target.b[i] += tau * (self.online.b[i] - self.target.b[i])


# ----------------------------------------------------------------------------------
# TRAINING
# ----------------------------------------------------------------------------------

def late_gold_delay(env: ClusterSchedulingEnv) -> float:
    """Mean delay of the late-arriving Gold cohort - the jobs where policies can differ."""
    late = [j for j in env.jobs if j.is_gold and j.arrival_time > 0]
    if not late:
        return float("nan")
    vals = [(j.scheduling_delay(env.now) if j.scheduled_time is not None
             else env.now - j.arrival_time) for j in late]
    return float(np.mean(vals))


def SELECTION_SCORE(ev: dict) -> float:
    """
    Model-selection score, lower is better. Mirrors the reward's own priorities:

        10 * mean_gold_delay + 1 * mean_bronze_delay

    Both metrics already charge unscheduled jobs their censored wait, so a policy that
    abandons work is penalised automatically rather than needing a separate term. Using
    gold delay alone would have selected the previous run's episode 20 - its best gold
    delay (810.5 s) came with Bronze collapsing to 5,108 s and 4,188 jobs unscheduled.
    """
    return (GOLD_PRIORITY_WEIGHT * ev["gold_delay"]
            + BRONZE_PRIORITY_WEIGHT * ev["bronze_wait"])


def greedy_eval(agent: "DQNAgent", seed: int = 0) -> dict:
    """
    One full episode with exploration off, instrumented with the same action counters as
    the training diagnostic.

    place_fail is the one to watch: a deterministic policy can lock onto a slot holding an
    unplaceable job and burn every step on a failed placement, which merely advances the
    clock. Exploration hides that failure mode during training (place_fail was 0 there),
    so it is only visible with epsilon switched off.
    """
    was_greedy = agent.greedy
    agent.greedy = True
    try:
        env = ClusterSchedulingEnv()
        obs, info = env.reset(seed=seed)
        mask = info["action_mask"]
        c = {"place_ok": 0, "place_fail": 0, "noop": 0}

        for _ in range(env.max_steps):
            action = agent.act(obs, mask)
            n_visible = len(env.visible_jobs())
            obs, _, terminated, truncated, info = env.step(action)
            mask = info["action_mask"]

            if action >= agent.n_actions - 1 or action >= n_visible:
                c["noop"] += 1
            elif info["scheduled"]:
                c["place_ok"] += 1
            else:
                c["place_fail"] += 1

            if terminated or truncated:
                break

        m = env.episode_metrics()
        return {"gold_delay": m["gold_avg_scheduling_delay"],
                "bronze_wait": m["bronze_avg_waiting_time"],
                "late_gold": late_gold_delay(env),
                "unscheduled": m["n_unscheduled"],
                "steps": m["steps"], **c}
    finally:
        agent.greedy = was_greedy


def train(n_episodes: int = N_EPISODES, seed: int = SEED, eval_every: int = EVAL_EVERY) -> tuple[DQNAgent, pd.DataFrame]:
    env = ClusterSchedulingEnv()  # tick=1.0 by default, core reward by default
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, seed=seed)
    log = TrainingLog()
    eval_rows: list[dict] = []
    best = {"score": float("inf"), "episode": None, "weights": None, "metrics": None}

    print(f"Environment : {env.n_jobs:,} jobs ({env.n_gold_total:,} Gold), "
          f"{env.n_machines} machines, tick={env.scheduling_tick}")
    print(f"Network     : {env.observation_space.shape[0]} -> {HIDDEN} -> {env.action_space.n}")
    print(f"Training    : {n_episodes} episodes\n")
    print(f"{'ep':>3s} {'steps':>7s} {'reward':>10s} {'R_gold':>9s} {'R_bronze':>9s} "
          f"{'R_queue':>9s} {'G:B':>7s} {'gold_delay':>10s} {'late_gold':>9s} "
          f"{'bronze_wait':>11s} {'eps':>5s} {'|TD|':>7s} {'mean|Q|':>8s} {'sec':>5s}")
    print("-" * 134)

    for ep in range(1, n_episodes + 1):
        t0 = time.time()
        obs, info = env.reset(seed=seed + ep)
        mask = info["action_mask"].copy()

        ep_reward = 0.0
        comp = {"util": 0.0, "gold": 0.0, "bronze": 0.0, "queue": 0.0, "noop": 0.0}
        losses = []
        qmags = []

        while True:
            action = agent.act(obs, mask)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_mask = info["action_mask"].copy()
            done = terminated or truncated

            shaped = float(np.clip(reward * REWARD_SCALE, -REWARD_CLIP, REWARD_CLIP))
            agent.buffer.add(obs, action, shaped, next_obs, float(terminated), next_mask)
            ep_reward += reward
            for key, val in info.get("reward_components", {}).items():
                comp[key] += val

            agent.total_steps += 1
            if agent.total_steps % TRAIN_EVERY == 0:
                loss = agent.train_step()
                if np.isfinite(loss):
                    losses.append(loss)
                    qmags.append(agent.last_q_abs)
                    agent.soft_update()

            obs, mask = next_obs, next_mask
            if done:
                break

        m = env.episode_metrics()
        lg = late_gold_delay(env)
        elapsed = time.time() - t0

        log.episode.append(ep)
        log.total_reward.append(ep_reward)
        log.gold_delay.append(m["gold_avg_scheduling_delay"])
        log.late_gold_delay.append(lg)
        log.bronze_wait.append(m["bronze_avg_waiting_time"])
        log.cpu_util.append(m["cpu_utilization"])
        log.epsilon.append(agent.epsilon())
        log.r_gold.append(comp["gold"])
        log.r_bronze.append(comp["bronze"])
        log.r_util.append(comp["util"])
        log.r_noop.append(comp["noop"])
        log.r_queue.append(comp["queue"])
        log.loss.append(float(np.mean(losses)) if losses else float("nan"))
        log.q_mag.append(float(np.mean(qmags)) if qmags else float("nan"))
        log.steps.append(m["steps"])
        log.wall_time.append(elapsed)

        if eval_every and (ep % eval_every == 0 or ep == 1):
            ev = greedy_eval(agent, seed=seed)
            print(f"    >>> GREEDY CHECKPOINT ep {ep:3d} | gold_delay {ev['gold_delay']:9,.1f} "
                  f"(SP 745.6) | bronze_wait {ev['bronze_wait']:9,.1f} (SP 1,018.9) | "
                  f"late_gold {ev['late_gold']:8,.1f} (SP 0.4) | unscheduled {ev['unscheduled']:6,d} | "
                  f"place_ok {ev['place_ok']:6,d} place_FAIL {ev['place_fail']:6,d} "
                  f"noop {ev['noop']:6,d}", flush=True)
            eval_rows.append({"episode": ep, **ev})

            score = SELECTION_SCORE(ev)
            if score < best["score"]:
                best.update(score=score, episode=ep, metrics=ev,
                            weights=([w.copy() for w in agent.online.W],
                                     [b.copy() for b in agent.online.b]))
                print(f"        (new best, score {score:,.1f})", flush=True)
            else:
                print(f"        (score {score:,.1f}; best remains ep {best['episode']} "
                      f"at {best['score']:,.1f})", flush=True)

        ratio = abs(comp["gold"]) / max(abs(comp["bronze"]), 1e-9)
        print(f"{ep:3d} {m['steps']:7,d} {ep_reward:10,.1f} {comp['gold']:9,.1f} "
              f"{comp['bronze']:9,.1f} {comp['queue']:9,.1f} {ratio:6.2f}x "
              f"{m['gold_avg_scheduling_delay']:10,.2f} {lg:9,.2f} "
              f"{m['bronze_avg_waiting_time']:11,.1f} {agent.epsilon():5.2f} "
              f"{log.loss[-1]:7.4f} {log.q_mag[-1]:8.4f} {elapsed:5.0f}", flush=True)

    if eval_rows:
        pd.DataFrame(eval_rows).to_csv(CSV_DIR / "dqn_greedy_checkpoints.csv", index=False)

    # CHECKPOINT-BEST SELECTION: ship the best greedy evaluation, not the last episode.
    # The previous run made this concrete - episode 40 scored 0 unscheduled with Bronze at
    # 1,104 s, then episode 50 collapsed to 18,683 unscheduled, and the final-episode
    # weights are what got evaluated and saved.
    if best["weights"] is not None:
        agent.online.W = [w.copy() for w in best["weights"][0]]
        agent.online.b = [b.copy() for b in best["weights"][1]]
        agent.target.copy_from(agent.online)
        print(f"\nCHECKPOINT-BEST SELECTION: restored episode {best['episode']} "
              f"(score {best['score']:,.1f})")
        print(f"  gold_delay {best['metrics']['gold_delay']:,.1f}  "
              f"bronze_wait {best['metrics']['bronze_wait']:,.1f}  "
              f"late_gold {best['metrics']['late_gold']:,.1f}  "
              f"unscheduled {best['metrics']['unscheduled']:,d}")
        rejected = [r for r in eval_rows if r["episode"] != best["episode"]]
        if rejected:
            worst = max(rejected, key=lambda r: SELECTION_SCORE(r))
            print(f"  (rejected episode {worst['episode']}: score "
                  f"{SELECTION_SCORE(worst):,.1f}, unscheduled {worst['unscheduled']:,d})")

    return agent, log.to_frame()


# ----------------------------------------------------------------------------------
# EVALUATION
# ----------------------------------------------------------------------------------

def evaluate(agent: DQNAgent, seed: int = 0) -> pd.DataFrame:
    """Greedy DQN against all four baselines on the same full environment."""
    agent.greedy = True
    rows = []

    policies = [FCFSScheduler(), RoundRobinScheduler(), StaticPriorityScheduler(),
                ResourceReservationScheduler(), agent]
    for policy in policies:
        env = ClusterSchedulingEnv()
        m = run_episode(env, policy, seed=seed)
        m["late_gold_avg_delay"] = late_gold_delay(env)
        late = [j for j in env.jobs if j.is_gold and j.arrival_time > 0]
        m["late_gold_scheduled"] = sum(1 for j in late if j.scheduled_time is not None)
        rows.append(m)
        print(f"  {m['policy']:20s} gold_delay={m['gold_avg_scheduling_delay']:9,.2f}  "
              f"late_gold={m['late_gold_avg_delay']:8,.2f}  "
              f"bronze_wait={m['bronze_avg_waiting_time']:9,.1f}  "
              f"cpu={m['cpu_utilization']:.4f}", flush=True)

    agent.greedy = False
    return pd.DataFrame(rows).set_index("policy")


def sanity_checks(agent: DQNAgent) -> list[tuple[str, bool, str]]:
    """Post-training checks specific to the agent, on top of test_environment.py."""
    results = []

    # 1. The greedy policy must never emit an illegal action.
    agent.greedy = True
    env = ClusterSchedulingEnv()
    obs, info = env.reset(seed=123)
    mask = info["action_mask"]
    illegal = 0
    for _ in range(20_000):
        a = agent.act(obs, mask)
        if not mask[a]:
            illegal += 1
        obs, _, term, trunc, info = env.step(a)
        mask = info["action_mask"]
        if term or trunc:
            break
    results.append(("DQN never selects a masked-out action", illegal == 0,
                    f"{illegal} illegal actions in {env.steps:,} steps"))

    # 2. Determinism: greedy policy must be reproducible.
    m1 = run_episode(ClusterSchedulingEnv(), agent, seed=7)
    m2 = run_episode(ClusterSchedulingEnv(), agent, seed=7)
    same = abs(m1["gold_avg_scheduling_delay"] - m2["gold_avg_scheduling_delay"]) < 1e-9
    results.append(("Greedy policy is deterministic", same,
                    f"{m1['gold_avg_scheduling_delay']:.6f} vs {m2['gold_avg_scheduling_delay']:.6f}"))

    # 3. Q-values must be finite.
    q = agent.online.forward(np.zeros((1, env.observation_space.shape[0]), dtype=np.float32))
    results.append(("Q-values are finite", bool(np.all(np.isfinite(q))), f"range {q.min():.3f}..{q.max():.3f}"))

    agent.greedy = False
    return results


def plot_training(log: pd.DataFrame, path=None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    axes[0, 0].plot(log["episode"], log["total_reward"], marker="o", ms=3, color="#3A6EA5")
    axes[0, 0].set_title("Episode return")
    axes[0, 0].set_xlabel("episode")

    axes[0, 1].plot(log["episode"], log["gold_delay"], marker="o", ms=3,
                    color="#C8963E", label="all Gold")
    axes[0, 1].plot(log["episode"], log["late_gold_delay"], marker="s", ms=3,
                    color="#B4432F", label="late Gold")
    axes[0, 1].set_yscale("symlog")
    axes[0, 1].set_title("Gold scheduling delay (s) - lower is better")
    axes[0, 1].set_xlabel("episode")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(log["episode"], log["bronze_wait"], marker="o", ms=3, color="#8C6A4A")
    axes[1, 0].set_title("Bronze average waiting time (s)")
    axes[1, 0].set_xlabel("episode")

    ax = axes[1, 1]
    ax.plot(log["episode"], log["epsilon"], color="#666666", label="epsilon")
    ax.set_xlabel("episode")
    ax.set_ylabel("epsilon")
    ax2 = ax.twinx()
    ax2.plot(log["episode"], log["loss"], color="#3A6EA5", label="mean |TD error|")
    ax2.set_ylabel("mean |TD error|")
    ax.set_title("Exploration and TD error")

    for a in axes.flat:
        a.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = path or (PLOT_DIR / "dqn_training.png")
    plt.savefig(path, dpi=120)
    print(f"\nTraining curves written to {path}")


def main() -> None:
    import sys
    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else N_EPISODES
    smoke = n_ep < N_EPISODES

    ensure_output_dirs()
    print("=" * 96)
    print(f"DQN TRAINING - priority-aware cluster scheduler ({n_ep} episodes)")
    print("=" * 96)

    agent, log = train(n_episodes=n_ep)

    if smoke:
        print("\nSMOKE TEST - reward balance check")
        print("-" * 60)
        g, b = log["r_gold"].abs().mean(), log["r_bronze"].abs().mean()
        print(f"  mean |Gold contribution|   : {g:12,.1f}")
        print(f"  mean |Bronze contribution| : {b:12,.1f}")
        print(f"  Gold:Bronze ratio          : {g / max(b, 1e-9):12,.2f}x "
              f"(target ~10x, was ~0.10x before the fix)")
        print(f"  mean |TD error| first 5 ep : {log['loss'].head(5).mean():12,.4f}")
        print(f"  mean |TD error| last 5 ep  : {log['loss'].tail(5).mean():12,.4f}")
        print(f"  mean |Q|       first 5 ep  : {log['q_mag'].head(5).mean():12,.4f}")
        print(f"  mean |Q|       last 5 ep   : {log['q_mag'].tail(5).mean():12,.4f}")
        td_growth = log['loss'].tail(5).mean() / max(log['loss'].head(5).mean(), 1e-9)
        q_growth = log['q_mag'].tail(5).mean() / max(log['q_mag'].head(5).mean(), 1e-9)
        print(f"  |TD| growth factor         : {td_growth:12,.2f}x")
        print(f"  |Q|  growth factor         : {q_growth:12,.2f}x")
        print("\n  Reading: if |Q| grows with |TD|, the value function is diverging.")
        print("  If |Q| is flat while |TD| climbs, the state distribution is widening.")
        print("\n  POLICY-HEALTH SIGNALS (TD error alone missed the stalling failure):")
        print(f"    steps/episode  first 5 -> last 5 : "
              f"{log['steps'].head(5).mean():10,.0f} -> {log['steps'].tail(5).mean():10,.0f}")
        print(f"    bronze wait    first 5 -> last 5 : "
              f"{log['bronze_wait'].head(5).mean():10,.1f} -> {log['bronze_wait'].tail(5).mean():10,.1f}")
        print(f"    gold delay     first 5 -> last 5 : "
              f"{log['gold_delay'].head(5).mean():10,.1f} -> {log['gold_delay'].tail(5).mean():10,.1f}")
        log.to_csv(CSV_DIR / "dqn_smoke_log.csv", index=False)
        print("\n  Smoke log written to dqn_smoke_log.csv")
        return
    log.to_csv(CSV_DIR / "dqn_training_log.csv", index=False)
    plot_training(log)

    print("\n" + "=" * 96)
    print("EVALUATION - greedy DQN vs baselines (full environment, tick=1.0)")
    print("=" * 96)
    results = evaluate(agent)
    results.to_csv(CSV_DIR / "dqn_evaluation.csv")

    cols = ["gold_avg_scheduling_delay", "late_gold_avg_delay", "late_gold_scheduled",
            "gold_sla_violation_rate", "bronze_avg_waiting_time",
            "bronze_avg_completion_time", "cpu_utilization", "n_gold_unscheduled",
            "n_bronze_scheduled", "throughput"]
    with pd.option_context("display.width", 200, "display.max_columns", 30,
                           "display.float_format", lambda v: f"{v:,.4f}"):
        print("\n" + "=" * 96)
        print("FINAL COMPARISON")
        print("=" * 96)
        print(results[cols].T)

    print("\n" + "=" * 96)
    print("POST-TRAINING SANITY CHECKS")
    print("=" * 96)
    for name, ok, detail in sanity_checks(agent):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:45s} {detail}")

    np.savez(MODEL_DIR / "dqn_weights.npz",
             **{f"W{i}": w for i, w in enumerate(agent.online.W)},
             **{f"b{i}": b for i, b in enumerate(agent.online.b)})
    print(f"\nWeights written to {MODEL_DIR / 'dqn_weights.npz'}")


if __name__ == "__main__":
    main()
