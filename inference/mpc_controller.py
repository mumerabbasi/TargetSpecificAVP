"""MPC controller for vehicle pursuit.

This module implements a Model Predictive Controller using a Kinematic
Bicycle Model for autonomous vehicle pursuit scenarios.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .config import InferenceConfig


@dataclass
class VehicleState:
    """Current state of the ego vehicle.

    Attributes:
        speed: Current speed in m/s.
        throttle: Current throttle value [0, 1].
        steer: Current steering angle [-1, 1].
        brake: Current brake value [0, 1].
    """

    speed: float = 0.0
    throttle: float = 0.0
    steer: float = 0.0
    brake: float = 0.0


@dataclass
class TargetPose:
    """Estimated pose of the target vehicle relative to ego.

    CARLA coordinate system:
        +x: Forward
        +y: Right
        yaw: Counter-clockwise positive (heading difference)

    Attributes:
        dx: Longitudinal distance in meters (+x forward).
        dy: Lateral offset in meters (+y right).
        dyaw: Relative yaw in degrees.
        target_speed: Estimated speed of target in m/s (optional).
    """

    dx: float = 0.0
    dy: float = 0.0
    dyaw: float = 0.0
    target_speed: float = 0.0


@dataclass
class ControlCommand:
    """Control command for the ego vehicle.

    Attributes:
        throttle: Throttle value [0, 1].
        steer: Steering angle [-1, 1], positive = right.
        brake: Brake value [0, 1].
    """

    throttle: float = 0.0
    steer: float = 0.0
    brake: float = 0.0


class MPCController:
    """MPC-based controller using Kinematic Bicycle Model.

    Uses a non-linear optimization solver (SLSQP) to find the optimal
    control sequence (acceleration and steering) that minimizes the
    cost function over a prediction horizon.

    Kinematic Bicycle Model:
        $$x_{t+1} = x_t + v_t \\cos(\\psi_t) dt$$
        $$y_{t+1} = y_t + v_t \\sin(\\psi_t) dt$$
        $$\\psi_{t+1} = \\psi_t + \\frac{v_t}{L} \\tan(\\delta_t) dt$$
        $$v_{t+1} = v_t + a_t dt$$

    Attributes:
        config: Inference configuration.
        dt: Time step for prediction in seconds.
        horizon: Number of steps to predict into the future.
        wheelbase: Distance between front and rear axles (L).
        u_prev: Previous control solution (used for warm-starting).
        prev_steer_cmd: Previous steering command for low-pass filtering.
    """

    def __init__(self, config: InferenceConfig) -> None:
        """Initialize the MPC controller.

        Args:
            config: Inference configuration.
        """
        self.config = config

        # MPC Parameters (from config)
        self.dt = config.mpc_dt
        self.horizon = config.mpc_horizon
        self.wheelbase = config.wheelbase

        # Constraints from config
        self.max_steer_rad = config.max_steer_rad
        self.max_accel = config.max_accel
        self.max_decel = config.max_decel

        # Cost weights from config
        self.w_dist = config.w_dist
        self.w_lat = config.w_lat
        self.w_yaw = config.w_yaw
        self.w_vel = config.w_vel
        self.w_steer = config.w_steer
        self.w_accel = config.w_accel
        self.w_dsteer = config.w_dsteer
        self.w_daccel = config.w_daccel

        # Low-pass filter coefficient from config
        self.steer_filter_alpha = config.steer_filter_alpha

        # Warm start storage (flat array: [acc_0, steer_0, acc_1, steer_1, ...])
        self.u_prev = np.zeros(self.horizon * 2)

        # Previous steering command for filtering
        self.prev_steer_cmd = 0.0

    def compute_control(
        self,
        target_pose: TargetPose,
        vehicle_state: VehicleState,
    ) -> ControlCommand:
        """Compute control command using optimization.

        Args:
            target_pose: Estimated pose of target vehicle.
            vehicle_state: Current state of ego vehicle.

        Returns:
            Control command for ego vehicle.
        """
        # Use target velocity from pose (from CARLA actor)
        target_velocity = target_pose.target_speed

        # Current State: [x, y, psi, v]
        x0 = np.array([0.0, 0.0, 0.0, vehicle_state.speed])

        # Convert dy from CARLA frame (Right+) to standard frame (Left+)
        target_y_std = -target_pose.dy
        target_yaw_rad = math.radians(target_pose.dyaw)

        # Reference position: desired distance behind target
        ref_x = target_pose.dx - self.config.desired_distance

        # Reference velocity: match target speed
        ref_velocity = target_velocity

        # Reference vector [x, y, psi, v]
        ref_state = np.array([ref_x, target_y_std, target_yaw_rad, ref_velocity])

        # Define bounds for optimization
        bounds = []
        for _ in range(self.horizon):
            bounds.append((self.max_decel, self.max_accel))
            bounds.append((-self.max_steer_rad, self.max_steer_rad))

        # Warm start: shift previous solution
        u_init = np.roll(self.u_prev, -2)
        u_init[-2:] = 0.0

        result = minimize(
            fun=self._cost_function,
            x0=u_init,
            args=(x0, ref_state),
            method="SLSQP",
            bounds=bounds,
            options={"ftol": 1e-3, "disp": False, "maxiter": 30},
        )

        # Extract optimal control actions
        self.u_prev = result.x
        optimal_accel = result.x[0]
        optimal_steer_rad = result.x[1]

        # Convert steering to CARLA frame [-1, 1]
        # Internal: +Steer = Left, CARLA: +Steer = Right
        raw_steer_cmd = -(optimal_steer_rad / self.max_steer_rad)
        raw_steer_cmd = np.clip(raw_steer_cmd, -1.0, 1.0)

        # Apply low-pass filter to steering for smoothness
        steer_cmd = (
            self.steer_filter_alpha * self.prev_steer_cmd
            + (1 - self.steer_filter_alpha) * raw_steer_cmd
        )
        self.prev_steer_cmd = steer_cmd

        # Convert acceleration to throttle/brake
        throttle_cmd = 0.0
        brake_cmd = 0.0

        if optimal_accel > 0:
            throttle_cmd = optimal_accel / self.max_accel
        else:
            brake_cmd = abs(optimal_accel) / abs(self.max_decel)

        # Apply safety limits
        throttle_cmd = float(np.clip(throttle_cmd, 0.0, self.config.max_throttle))
        brake_cmd = float(np.clip(brake_cmd, 0.0, self.config.max_brake))
        steer_cmd = float(np.clip(steer_cmd, -self.config.max_steer, self.config.max_steer))

        # Emergency brake if too close
        if target_pose.dx < self.config.collision_distance:
            return ControlCommand(throttle=0.0, steer=steer_cmd, brake=1.0)

        # Gradual slowdown if approaching collision distance
        if target_pose.dx < self.config.slowdown_distance:
            slowdown_factor = (
                (target_pose.dx - self.config.collision_distance)
                / (self.config.slowdown_distance - self.config.collision_distance)
            )
            throttle_cmd *= slowdown_factor
            brake_cmd = max(brake_cmd, 0.3 * (1 - slowdown_factor))

        return ControlCommand(
            throttle=throttle_cmd,
            steer=steer_cmd,
            brake=brake_cmd,
        )

    def _bicycle_model(
        self,
        state: np.ndarray,
        control: np.ndarray,
    ) -> np.ndarray:
        """Apply Kinematic Bicycle Model dynamics.

        Args:
            state: [x, y, psi, v]
            control: [accel, steer_angle]

        Returns:
            Next state [x, y, psi, v]
        """
        x, y, psi, v = state
        a, delta = control

        # Kinematic bicycle model update
        next_x = x + v * math.cos(psi) * self.dt
        next_y = y + v * math.sin(psi) * self.dt
        next_psi = psi + (v / self.wheelbase) * math.tan(delta) * self.dt
        next_v = v + a * self.dt

        # Prevent reversing
        next_v = max(0.0, next_v)

        return np.array([next_x, next_y, next_psi, next_v])

    def _cost_function(
        self,
        u_flat: np.ndarray,
        x0: np.ndarray,
        ref: np.ndarray,
    ) -> float:
        """Calculate total cost for the control sequence.

        Args:
            u_flat: Flattened control array [a0, s0, a1, s1, ...].
            x0: Initial state.
            ref: Reference target state [ref_x, ref_y, ref_psi, ref_v].

        Returns:
            Scalar cost.
        """
        cost = 0.0
        current_state = x0.copy()

        # Reshape controls to (Horizon, 2)
        controls = u_flat.reshape((self.horizon, 2))

        prev_u = np.zeros(2)

        for t in range(self.horizon):
            u_t = controls[t]

            # Update state using bicycle model
            current_state = self._bicycle_model(current_state, u_t)

            # Calculate tracking errors
            e_dx = current_state[0] - ref[0]
            e_dy = current_state[1] - ref[1]
            e_psi = current_state[2] - ref[2]
            e_v = current_state[3] - ref[3]

            # Normalize yaw error to [-pi, pi]
            e_psi = (e_psi + np.pi) % (2 * np.pi) - np.pi

            # State costs
            cost += self.w_dist * (e_dx ** 2)
            cost += self.w_lat * (e_dy ** 2)
            cost += self.w_yaw * (e_psi ** 2)
            cost += self.w_vel * (e_v ** 2)  # Velocity matching cost

            # Control costs (input minimization)
            cost += self.w_accel * (u_t[0] ** 2)
            cost += self.w_steer * (u_t[1] ** 2)

            # Smoothness costs (change rate minimization)
            if t > 0:
                d_accel = u_t[0] - prev_u[0]
                d_steer = u_t[1] - prev_u[1]
                cost += self.w_daccel * (d_accel ** 2)
                cost += self.w_dsteer * (d_steer ** 2)

            prev_u = u_t

        return cost

    def reset(self) -> None:
        """Reset controller state."""
        self.u_prev = np.zeros(self.horizon * 2)
        self.prev_steer_cmd = 0.0
