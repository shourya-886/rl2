"""
Gymnasium environment for the rotary inverted pendulum (Furuta pendulum)
defined in inverted_pendulum.xml.

Two hinge joints:
  - base_to_arm     : actuated (position servo), horizontal rotation, limited
                       to +-135 deg (2.35619449 rad) by hardware stops.
  - arm_to_pendulum : unactuated, free-spinning, this is the joint that must
                       be swung up and balanced.

Task: swing the pendulum up from hanging straight down (angle = pi, or
equivalently -pi) to balanced upright (angle = 0) and hold it there, by
only ever commanding the arm's target angle.

Requires: inverted_pendulum.xml (and its meshes/ dir) in the same directory,
or pass an explicit xml_path.
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco


class InvertedPendulumEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, xml_path=None):
        super().__init__()

        xml_path = xml_path or os.path.join(
            os.path.dirname(__file__), "inverted_pendulum.xml"
        )
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.frame_skip = 4          # sim dt in the XML is unset -> default 0.002s; 4*0.002=0.008s control dt
        self.max_steps = 1000
        self.upright_tolerance = 0.15   # rad, ~8.6 deg, counted as "balanced"
        self.upright_hold_steps = 100   # steps balanced before success termination

        # Joint / actuator indices, resolved once
        self.arm_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_to_arm")
        self.pend_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "arm_to_pendulum")
        self.arm_qpos_adr = self.model.jnt_qposadr[self.arm_joint_id]
        self.pend_qpos_adr = self.model.jnt_qposadr[self.pend_joint_id]
        self.arm_dof_adr = self.model.jnt_dofadr[self.arm_joint_id]
        self.pend_dof_adr = self.model.jnt_dofadr[self.pend_joint_id]

        self.arm_limit = 2.35619449  # rad, matches the XML's hardware stop range

        # --- Action: target arm angle (rad), fed straight to the position actuator ---
        self.action_space = spaces.Box(
            low=np.array([-self.arm_limit], dtype=np.float32),
            high=np.array([self.arm_limit], dtype=np.float32),
        )

        # --- Observation: [arm_angle, arm_vel, cos(pend), sin(pend), pend_vel] ---
        # cos/sin of the pendulum angle instead of the raw angle avoids the
        # wraparound discontinuity at +-pi (the hanging-down rest position),
        # which would otherwise sit right at the edge of the observation range.
        high = np.array([self.arm_limit, 30.0, 1.0, 1.0, 50.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.render_mode = render_mode
        self.step_count = 0
        self.balanced_steps = 0
        self._renderer = None

    def _pendulum_angle_wrapped(self):
        """Wrap raw pendulum qpos to [-pi, pi], with 0 = upright, +-pi = hanging down."""
        raw = self.data.qpos[self.pend_qpos_adr]
        return np.arctan2(np.sin(raw), np.cos(raw))

    def _get_obs(self):
        arm_angle = self.data.qpos[self.arm_qpos_adr]
        arm_vel = self.data.qvel[self.arm_dof_adr]
        pend_angle = self._pendulum_angle_wrapped()
        pend_vel = self.data.qvel[self.pend_dof_adr]
        return np.array(
            [arm_angle, arm_vel, np.cos(pend_angle), np.sin(pend_angle), pend_vel],
            dtype=np.float32,
        )

    def _get_info(self):
        return {
            "pendulum_angle": float(self._pendulum_angle_wrapped()),
            "arm_angle": float(self.data.qpos[self.arm_qpos_adr]),
            "balanced_steps": self.balanced_steps,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Start hanging down (pi) with a small random perturbation, arm centered.
        self.data.qpos[self.arm_qpos_adr] = self.np_random.uniform(-0.1, 0.1)
        self.data.qpos[self.pend_qpos_adr] = np.pi + self.np_random.uniform(-0.05, 0.05)
        self.data.ctrl[0] = 0.0

        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        self.balanced_steps = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        target_angle = float(np.clip(action[0], -self.arm_limit, self.arm_limit))
        self.data.ctrl[0] = target_angle

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, -50.0, True, False, {"diverged": True}

        self.step_count += 1
        obs = self._get_obs()
        arm_angle, arm_vel, cos_p, sin_p, pend_vel = obs
        pend_angle = np.arctan2(sin_p, cos_p)

        # Reward: upright (angle near 0) and slow (low angular velocity) is best.
        # cos_p alone ranges [-1, 1] -> 1 at upright, -1 hanging down; scale it
        # as the main shaping term, penalize angular velocity and large arm
        # excursions (keeps the arm from just spinning to fling the pendulum up).
        reward = cos_p - 0.01 * (pend_vel ** 2) - 0.001 * (arm_vel ** 2) - 0.01 * (arm_angle ** 2)

        is_balanced = abs(pend_angle) < self.upright_tolerance
        self.balanced_steps = self.balanced_steps + 1 if is_balanced else 0

        hit_arm_limit = abs(arm_angle) >= self.arm_limit - 1e-3
        success = self.balanced_steps >= self.upright_hold_steps
        truncated = self.step_count >= self.max_steps

        if success:
            reward += 50.0
        if hit_arm_limit:
            reward -= 5.0  # discourage slamming into hardware stops, but don't end the episode over it

        terminated = bool(success)

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, bool(truncated), self._get_info()

    def render(self):
        if self.render_mode is None:
            return
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data)
        frame = self._renderer.render()
        if self.render_mode == "rgb_array":
            return frame
        return frame  # "human": no live window headless; use mujoco.viewer.launch_passive locally

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


if __name__ == "__main__":
    env = InvertedPendulumEnv(xml_path="/home/shourya/rl2/urdf/model.xml")
    obs, info = env.reset(seed=0)
    total_reward = 0.0

    for i in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if i % 40 == 0:
            print(f"step={i} obs={np.round(obs, 3)} reward={reward:.3f} info={info}")
        if terminated or truncated:
            print("Episode ended:", info)
            break

    print(f"Total reward: {total_reward:.2f}")
    env.close()