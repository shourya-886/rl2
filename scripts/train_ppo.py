import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from inverted_pendulum_env import InvertedPendulumEnv

TOTAL_TIMESTEPS = 1_000_000
N_ENVS = 8
CHECKPOINT_FREQ = 50_000
EVAL_FREQ = 20_000
MODEL_DIR = "./models"
LOG_DIR = "./logs"


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.join(MODEL_DIR, "checkpoints"), exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    check_env(InvertedPendulumEnv())
    print("check_env passed.")

    train_env = make_vec_env(InvertedPendulumEnv, n_envs=N_ENVS)
    eval_env = make_vec_env(InvertedPendulumEnv, n_envs=1)

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,          # High exploration entropy to discover swing-up trajectory
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=1,
        tensorboard_log=LOG_DIR,
        device="auto",
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

    print(f"Starting training across {N_ENVS} parallel envs...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    final_path = os.path.join(MODEL_DIR, "ppo_pendulum_final")
    model.save(final_path)
    print(f"Training complete. Saved to {final_path}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()