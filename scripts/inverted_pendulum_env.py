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
commanding relative delta-position changes for the arm.

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

        self.frame_skip = 4          # sim dt default 0.002s; 4*0.002 = 0.008s control dt
        self.max_steps = 1000
        self.upright_tolerance = 0.15   # rad (~8.6 deg), counted as "balanced"
        self.upright_hold_steps = 625   # steps balanced before counted as success (625 * 0.008s = 5s)

        # Joint / actuator indices, resolved once
        self.arm_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "base_to_arm")
        self.pend_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "arm_to_pendulum")
        self.arm_qpos_adr = self.model.jnt_qposadr[self.arm_joint_id]
        self.pend_qpos_adr = self.model.jnt_qposadr[self.pend_joint_id]
        self.arm_dof_adr = self.model.jnt_dofadr[self.arm_joint_id]
        self.pend_dof_adr = self.model.jnt_dofadr[self.pend_joint_id]

        self.arm_limit = 2.35619449  # rad (+-135 deg hardware stop range)

        # --- Relative Action Settings ---
        self.max_delta = 0.10  # Max target angle delta per step (rad, ~5.7 deg)

        # --- Action: Normalized [-1.0, 1.0], representing delta movement ---
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # --- Observation: [arm_angle, arm_vel, cos(pend), sin(pend), pend_vel] ---
        high = np.array([self.arm_limit, 60.0, 1.0, 1.0, 150.0], dtype=np.float32)
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
        obs = np.array(
            [arm_angle, arm_vel, np.cos(pend_angle), np.sin(pend_angle), pend_vel],
            dtype=np.float32,
        )
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

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
        
        # Initialize target control position to current arm position
        self.data.ctrl[0] = self.data.qpos[self.arm_qpos_adr]

        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        self.balanced_steps = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        # --------------------------------------------------------
        # Action: Relative Delta-Position Control
        # --------------------------------------------------------
        a = float(np.clip(action[0], -1.0, 1.0))
        delta_theta = a * self.max_delta
        
        current_theta = float(self.data.qpos[self.arm_qpos_adr])
        target_angle = float(
            np.clip(current_theta + delta_theta, -self.arm_limit, self.arm_limit)
        )

        self.data.ctrl[0] = target_angle

        # Run MuJoCo simulation
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # --------------------------------------------------------
        # Check simulation stability
        # --------------------------------------------------------
        if (
            not np.all(np.isfinite(self.data.qpos))
            or not np.all(np.isfinite(self.data.qvel))
        ):
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, -50.0, True, False, {"diverged": True}

        self.step_count += 1

        # --------------------------------------------------------
        # State & Reward Computation
        # --------------------------------------------------------
        obs = self._get_obs()

        theta = float(self.data.qpos[self.arm_qpos_adr])
        theta_dot = float(self.data.qvel[self.arm_dof_adr])
        alpha = float(self._pendulum_angle_wrapped())
        alpha_dot = float(self.data.qvel[self.pend_dof_adr])

        # Continuous Shaped Reward:
        # Penalizes magnitude of action step 'a' instead of absolute target angle
        reward = (
            2.0 * np.cos(alpha)
            + 0.2 * np.cos(theta)
            - 0.005 * (alpha_dot ** 2)
            - 0.001 * (theta_dot ** 2)
            - 0.01 * (a ** 2)
        )

        # --------------------------------------------------------
        # Balanced & Terminal Detection
        # --------------------------------------------------------
        is_balanced = abs(alpha) < self.upright_tolerance
        self.balanced_steps = self.balanced_steps + 1 if is_balanced else 0

        hit_arm_limit = abs(theta) >= self.arm_limit - 1e-3
        just_reached_hold = self.balanced_steps == self.upright_hold_steps
        truncated = self.step_count >= self.max_steps

        # Additional environment rewards/penalties
        if just_reached_hold:
            reward += 50.0

        if hit_arm_limit:
            reward -= 5.0

        terminated = False

        # --------------------------------------------------------
        # Rendering
        # --------------------------------------------------------
        if self.render_mode == "human":
            self.render()

        return (
            obs,
            float(reward),
            terminated,
            bool(truncated),
            self._get_info()
        )

    def render(self):
        if self.render_mode is None:
            return
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data)
        frame = self._renderer.render()
        if self.render_mode == "rgb_array":
            return frame
        return frame

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


if __name__ == "__main__":
    env = InvertedPendulumEnv()
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