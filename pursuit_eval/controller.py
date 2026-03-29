"""Fresh MPC controller for target-vehicle pursuit."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .config import PursuitEvalConfig


@dataclass
class VehicleState:
    """Current ego state used by the controller."""

    speed_mps: float = 0.0
    throttle: float = 0.0
    steer: float = 0.0
    brake: float = 0.0


@dataclass
class RelativeTargetPose:
    """Relative target pose in the ego frame."""

    dx_m: float = 0.0
    dy_m: float = 0.0
    yaw_deg: float = 0.0
    target_speed_mps: float = 0.0


@dataclass
class ControlCommand:
    """Low-level control command."""

    throttle: float = 0.0
    steer: float = 0.0
    brake: float = 0.0


class MPCFollower:
    """Model predictive controller for pursuit following."""

    def __init__(self, config: PursuitEvalConfig) -> None:
        self.config = config
        self.dt = config.mpc_dt
        self.horizon = config.mpc_horizon
        self.wheelbase = config.wheelbase_m
        self.prev_solution = np.zeros(self.horizon * 2, dtype=np.float64)
        self.prev_steer_cmd = 0.0

    def reset(self) -> None:
        self.prev_solution = np.zeros(self.horizon * 2, dtype=np.float64)
        self.prev_steer_cmd = 0.0

    def compute_control(
        self,
        target_pose: RelativeTargetPose,
        vehicle_state: VehicleState,
    ) -> ControlCommand:
        """Solve one MPC step for the current target estimate."""
        x0 = np.array(
            [
                float(target_pose.dx_m),
                float(target_pose.dy_m),
                math.radians(float(target_pose.yaw_deg)),
                float(vehicle_state.speed_mps),
            ],
            dtype=np.float64,
        )
        target_speed = max(0.0, float(target_pose.target_speed_mps))
        ref = np.array(
            [
                float(self.config.desired_distance_m),
                0.0,
                0.0,
                target_speed,
            ],
            dtype=np.float64,
        )

        bounds = []
        for _ in range(self.horizon):
            bounds.append(
                (float(
                    self.config.max_decel), float(
                    self.config.max_accel)))
            bounds.append((-float(self.config.max_steer_rad),
                          float(self.config.max_steer_rad)))

        u_init = np.roll(self.prev_solution, -2)
        accel_seed = self._initial_accel_seed(target_pose, vehicle_state)
        steer_seed = self._initial_steer_seed(target_pose)
        if not np.any(np.abs(u_init) > 1e-6):
            for step in range(self.horizon):
                u_init[2 * step] = accel_seed
                u_init[2 * step + 1] = steer_seed
        else:
            u_init[-2] = accel_seed
            u_init[-1] = steer_seed

        init_cost = self._cost_function(u_init, x0, ref, target_speed)
        result = minimize(
            fun=self._cost_function,
            x0=u_init,
            args=(x0, ref, target_speed),
            method="SLSQP",
            bounds=bounds,
            options={"ftol": 1e-3, "disp": False, "maxiter": 60},
        )
        optimal = u_init
        if result.x is not None and np.all(np.isfinite(result.x)):
            result_cost = self._cost_function(result.x, x0, ref, target_speed)
            if result.success or result_cost < init_cost:
                optimal = result.x

        self.prev_solution = np.asarray(optimal, dtype=np.float64)
        accel = float(self.prev_solution[0])
        steer_rad = float(self.prev_solution[1])

        raw_steer_cmd = steer_rad / float(self.config.max_steer_rad)
        raw_steer_cmd = float(np.clip(raw_steer_cmd, -1.0, 1.0))
        steer_cmd = (
            float(self.config.steer_filter_alpha) * self.prev_steer_cmd
            + (1.0 - float(self.config.steer_filter_alpha)) * raw_steer_cmd
        )
        self.prev_steer_cmd = steer_cmd

        throttle = 0.0
        brake = 0.0
        if accel > 0.0:
            throttle = accel / float(self.config.max_accel)
        else:
            brake = abs(accel) / abs(float(self.config.max_decel))

        throttle = float(
            np.clip(
                throttle, 0.0, float(
                    self.config.max_throttle)))
        brake = float(np.clip(brake, 0.0, float(self.config.max_brake)))
        steer_cmd = float(
            np.clip(
                steer_cmd, -float(self.config.max_steer),
                float(self.config.max_steer)))

        if (throttle > 0.0 and float(vehicle_state.speed_mps)
                < float(self.config.launch_speed_threshold_mps)):
            throttle = max(throttle, float(self.config.launch_throttle_floor))

        if float(target_pose.dx_m) < float(self.config.collision_distance_m):
            return ControlCommand(throttle=0.0, steer=steer_cmd, brake=1.0)

        if float(target_pose.dx_m) < float(self.config.slowdown_distance_m):
            denom = max(float(self.config.slowdown_distance_m) -
                        float(self.config.collision_distance_m), 1e-6, )
            factor = (float(target_pose.dx_m) -
                      float(self.config.collision_distance_m)) / denom
            factor = float(np.clip(factor, 0.0, 1.0))
            throttle *= factor
            brake = max(brake, 0.3 * (1.0 - factor))

        return ControlCommand(throttle=throttle, steer=steer_cmd, brake=brake)

    def _initial_accel_seed(
        self,
        target_pose: RelativeTargetPose,
        vehicle_state: VehicleState,
    ) -> float:
        gap_error_m = float(target_pose.dx_m) - \
            float(self.config.desired_distance_m)
        speed_error_mps = float(
            target_pose.target_speed_mps) - float(vehicle_state.speed_mps)
        accel_seed = 0.8 * gap_error_m + 1.2 * speed_error_mps
        return float(
            np.clip(
                accel_seed, float(
                    self.config.max_decel), float(
                    self.config.max_accel)))

    def _initial_steer_seed(self, target_pose: RelativeTargetPose) -> float:
        yaw_term = math.radians(float(target_pose.yaw_deg))
        lateral_term = math.atan2(
            float(
                target_pose.dy_m), max(
                float(
                    target_pose.dx_m), 1e-3))
        steer_seed = yaw_term + 0.8 * lateral_term
        return float(
            np.clip(
                steer_seed, -float(self.config.max_steer_rad),
                float(self.config.max_steer_rad)))

    def _relative_model(
            self,
            state: np.ndarray,
            control: np.ndarray,
            target_speed_mps: float) -> np.ndarray:
        dx_m, dy_m, yaw_rel_rad, ego_speed_mps = state
        accel, steer = control
        ego_yaw_rate = 0.0
        if abs(self.wheelbase) > 1e-6:
            ego_yaw_rate = (ego_speed_mps / self.wheelbase) * math.tan(steer)

        next_dx = dx_m + (
            float(target_speed_mps) * math.cos(yaw_rel_rad)
            - ego_speed_mps
            + ego_yaw_rate * dy_m
        ) * self.dt
        next_dy = dy_m + (
            float(target_speed_mps) * math.sin(yaw_rel_rad)
            - ego_yaw_rate * dx_m
        ) * self.dt
        next_yaw = yaw_rel_rad - ego_yaw_rate * self.dt
        next_yaw = (next_yaw + math.pi) % (2.0 * math.pi) - math.pi
        next_speed = max(0.0, ego_speed_mps + accel * self.dt)
        return np.array([next_dx, next_dy, next_yaw,
                        next_speed], dtype=np.float64)

    def _cost_function(
        self,
        u_flat: np.ndarray,
        x0: np.ndarray,
        ref: np.ndarray,
        target_speed_mps: float,
    ) -> float:
        cost = 0.0
        state = x0.copy()
        controls = np.asarray(
            u_flat, dtype=np.float64).reshape(
            (self.horizon, 2))
        prev_u = np.zeros(2, dtype=np.float64)

        for step in range(self.horizon):
            u_t = controls[step]
            state = self._relative_model(state, u_t, target_speed_mps)
            e_dx = state[0] - ref[0]
            e_dy = state[1]
            e_yaw = state[2]
            e_speed = state[3] - ref[3]

            cost += float(self.config.w_dist) * (e_dx ** 2)
            cost += float(self.config.w_lat) * (e_dy ** 2)
            cost += float(self.config.w_yaw) * (e_yaw ** 2)
            cost += float(self.config.w_vel) * (e_speed ** 2)
            cost += float(self.config.w_accel) * (u_t[0] ** 2)
            cost += float(self.config.w_steer) * (u_t[1] ** 2)

            if step > 0:
                d_accel = u_t[0] - prev_u[0]
                d_steer = u_t[1] - prev_u[1]
                cost += float(self.config.w_daccel) * (d_accel ** 2)
                cost += float(self.config.w_dsteer) * (d_steer ** 2)
            prev_u = u_t

        return float(cost)
