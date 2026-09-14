import argparse
import time

import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

from inverted_pendulum_env import InvertedPendulumEnv

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


def evaluate_with_render(model_path, xml_path=None, n_episodes=5, deterministic=True, slow_factor=1.0):
    env = InvertedPendulumEnv(xml_path=xml_path)
    model = PPO.load(model_path)

    print(f"Loaded model: {model_path}")
    print(f"Running {n_episodes} episode(s)... Close viewer window to stop.\n")

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

                viewer.sync()
                time.sleep(env.model.opt.timestep * env.frame_skip * slow_factor)

                if terminated or truncated:
                    break

            if not viewer.is_running():
                print("Viewer closed.")
                break

            status = "SUCCESS (balanced)" if info.get("balanced_steps", 0) >= env.upright_hold_steps else "ended"
            print(f"Episode {ep + 1}/{n_episodes}: reward={episode_reward:.2f}, "
                  f"steps={step}, {status}, final_pendulum_angle={info.get('pendulum_angle', float('nan')):.3f} rad (Target: ~3.14 rad)")

    env.close()


def record_video(model_path, xml_path=None, n_episodes=1, video_path="pendulum_eval.mp4", deterministic=True, fps=50):
    if not HAS_IMAGEIO:
        raise ImportError("Install imageio via: pip install imageio[ffmpeg]")

    env = InvertedPendulumEnv(render_mode="rgb_array", xml_path=xml_path)
    model = PPO.load(model_path)

    print(f"Recording {n_episodes} episode(s) to '{video_path}'...")
    frames = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        step = 0

        while True:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1

            if terminated or truncated:
                break

        status = "SUCCESS (balanced)" if info.get("balanced_steps", 0) >= env.upright_hold_steps else "ended"
        print(f"Episode {ep + 1}/{n_episodes}: reward={episode_reward:.2f}, "
              f"steps={step}, {status}, final_pendulum_angle={info.get('pendulum_angle', float('nan')):.3f} rad")

    env.close()
    imageio.mimsave(video_path, frames, fps=fps)
    print(f"Video saved to '{video_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model.")
    parser.add_argument("--model", type=str, default="models/best/best_model.zip")
    parser.add_argument("--xml", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--slow", type=float, default=1.0)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--video-path", type=str, default="pendulum_eval.mp4")
    args = parser.parse_args()

    if args.record:
        record_video(
            model_path=args.model,
            xml_path=args.xml,
            n_episodes=args.episodes,
            video_path=args.video_path,
            deterministic=not args.stochastic,
        )
    else:
        evaluate_with_render(
            model_path=args.model,
            xml_path=args.xml,
            n_episodes=args.episodes,
            deterministic=not args.stochastic,
            slow_factor=args.slow,
        )