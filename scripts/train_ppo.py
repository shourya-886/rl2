"""
Train PPO on InvertedPendulumEnv for 1,000,000 timesteps.

Usage:
    python train_ppo.py

Outputs:
    ./models/ppo_pendulum_final.zip   - final trained policy
    ./models/checkpoints/             - periodic checkpoints during training
    ./logs/                           - TensorBoard logs (view with `tensorboard --logdir logs`)

Requires: stable-baselines3, and inverted_pendulum_env.py + inverted_pendulum.xml
(+ meshes/) in the same directory as this script.
"""

import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from inverted_pendulum_env import InvertedPendulumEnv

TOTAL_TIMESTEPS = 1_000_000
N_ENVS = 8                     # parallel envs -- reduce if your CPU has fewer cores
CHECKPOINT_FREQ = 50_000       # per-env step count between checkpoints
EVAL_FREQ = 20_000
MODEL_DIR = "./models"
LOG_DIR = "./logs"


def make_env():
    """Factory so each of the N_ENVS parallel workers gets its own env instance."""
    def _init():
        env = InvertedPendulumEnv()
        env = Monitor(env)  # tracks episode reward/length for logging
        return env
    return _init


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.join(MODEL_DIR, "checkpoints"), exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Sanity-check the env against the Gymnasium API contract once, before
    # spending any real training compute -- catches shape/dtype/bounds bugs early.
    check_env(InvertedPendulumEnv())
    print("check_env passed.")

    # Vectorized envs: N_ENVS copies stepped in parallel, one physics sim each.
    # This is the standard way to get PPO the large, diverse rollout batches
    # it needs -- one single env stepped 1M times would be far slower and
    # give much more correlated (less diverse) experience per update.
    train_env = make_vec_env(make_env(), n_envs=N_ENVS)

    # Separate env (not vectorized) purely for periodic evaluation, so
    # eval performance isn't affected by whatever the training envs are
    # currently doing mid-episode.
    eval_env = make_vec_env(make_env(), n_envs=1)

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,          # rollout length per env before each policy update
        batch_size=256,
        n_epochs=10,
        gamma=0.99,            # discount factor
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,          # bump this up (e.g. 0.01) if training collapses to a bad local optimum
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="auto",         # uses GPU if available, else CPU -- MLPs this small rarely benefit much from GPU
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(CHECKPOINT_FREQ // N_ENVS, 1),
        save_path=os.path.join(MODEL_DIR, "checkpoints"),
        name_prefix="ppo_pendulum",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODEL_DIR, "best"),
        log_path=LOG_DIR,
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes=5,
        deterministic=True,
    )

    print(f"Starting training for {TOTAL_TIMESTEPS:,} timesteps across {N_ENVS} parallel envs...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    final_path = os.path.join(MODEL_DIR, "ppo_pendulum_final")
    model.save(final_path)
    print(f"Training complete. Final model saved to {final_path}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()