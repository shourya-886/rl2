"""
Load a trained PPO policy and watch it run in a live MuJoCo viewer window.

Usage:
    python evaluate_render.py
    python evaluate_render.py --model models/best/best_model.zip
    python evaluate_render.py --model models/ppo_pendulum_final.zip --episodes 5
    python evaluate_render.py --xml /home/shourya/rl2/scripts/inverted_pendulum.xml

Controls in the viewer window (standard MuJoCo viewer):
    - Left-drag: rotate camera
    - Right-drag / scroll: pan / zoom
    - Space: pause/unpause physics
    - Esc or close window: quit

This must be run on a machine with an actual display (your laptop/PC) --
it will not open a window in a headless environment.
"""

import argparse
import time

import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

from inverted_pendulum_env import InvertedPendulumEnv


def evaluate_with_render(model_path, xml_path=None, n_episodes=5, deterministic=True, slow_factor=1.0):
    env = InvertedPendulumEnv(xml_path=xml_path)
    model = PPO.load(model_path)

    print(f"Loaded model from {model_path}")
    print(f"Running {n_episodes} episode(s). Close the viewer window or Ctrl+C to stop early.\n")

    # launch_passive gives us a window we drive manually (step physics, then
    # viewer.sync()) -- the alternative, launch(), blocks and runs its own
    # loop, which doesn't let us inject the policy's actions per step.
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for ep in range(n_episodes):
            obs, info = env.reset()
            episode_reward = 0.0
            step = 0

            while viewer.is_running():
                action, _states = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step += 1

                viewer.sync()  # push the current physics state to the window
                # env.frame_skip * model timestep = real seconds per env.step();
                # sleeping keeps playback roughly real-time instead of running
                # as fast as the CPU can go. Increase slow_factor to watch in
                # slow motion, e.g. slow_factor=3.0.
                time.sleep(env.model.opt.timestep * env.frame_skip * slow_factor)

                if terminated or truncated:
                    break

            status = "SUCCESS (balanced)" if info.get("balanced_steps", 0) >= env.upright_hold_steps else "ended"
            print(f"Episode {ep + 1}/{n_episodes}: reward={episode_reward:.2f}, "
                  f"steps={step}, {status}, final_pendulum_angle={info.get('pendulum_angle', float('nan')):.3f} rad")

            if not viewer.is_running():
                print("Viewer window closed, stopping.")
                break

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model with live MuJoCo rendering.")
    parser.add_argument("--model", type=str, default="models/ppo_pendulum_final.zip",
                         help="Path to the trained SB3 model .zip file.")
    parser.add_argument("--xml", type=str, default=None,
                         help="Path to the MuJoCo XML (defaults to inverted_pendulum.xml next to the env file).")
    parser.add_argument("--episodes", type=int, default=5,
                         help="Number of episodes to run.")
    parser.add_argument("--stochastic", action="store_true",
                         help="Sample actions stochastically instead of using the deterministic policy mean.")
    parser.add_argument("--slow", type=float, default=1.0,
                         help="Slow-motion factor (e.g. 3.0 = 3x slower than real time).")
    args = parser.parse_args()

    evaluate_with_render(
        model_path=args.model,
        xml_path=args.xml,
        n_episodes=args.episodes,
        deterministic=not args.stochastic,
        slow_factor=args.slow,
    )