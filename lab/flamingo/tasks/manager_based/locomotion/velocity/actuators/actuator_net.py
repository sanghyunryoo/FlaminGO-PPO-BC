import torch
from collections.abc import Sequence
from isaaclab.utils.assets import read_file

from isaaclab.utils.types import ArticulationActions

from isaaclab.actuators.actuator_net import (
    ActuatorNetLSTM as BaseActuatorNetLSTM,
    ActuatorNetMLP as BaseActuatorNetMLP,
)

from isaaclab.actuators.actuator_pd import DCMotor, DelayedPDActuator


class ActuatorNetLSTM(BaseActuatorNetLSTM):
    """Extended Actuator model based on recurrent neural network (LSTM)."""

    def __init__(self, cfg, *args, **kwargs):
        # Delayed import to avoid circular dependency
        from .actuator_cfg import ActuatorNetLSTMCfg

        self.cfg: ActuatorNetLSTMCfg = cfg
        super().__init__(cfg, *args, **kwargs)

        # Additional initializations or overrides if necessary
        if self.cfg.input_order not in ["pos_vel", "vel_pos"]:
            raise ValueError(
                f"Invalid input order for LSTM actuator net: {self.cfg.input_order}. Must be 'pos_vel' or 'vel_pos'."
            )

        input_dim = 2 if self.cfg.input_order == "pos_vel" else 3
        self.sea_input = torch.zeros(self._num_envs * self.num_joints, 1, input_dim, device=self._device)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        if self.cfg.input_order == "pos_vel":
            self.sea_input[:, 0, 0] = (control_action.joint_positions - joint_pos).flatten()
            self.sea_input[:, 0, 1] = joint_vel.flatten()
        elif self.cfg.input_order == "vel_pos":
            self.sea_input[:, 0, 0] = (control_action.joint_velocities - joint_vel).flatten()
            self.sea_input[:, 0, 1] = torch.sin(joint_pos).flatten()
            self.sea_input[:, 0, 2] = torch.cos(joint_pos).flatten()

        self._joint_vel[:] = joint_vel

        with torch.inference_mode():
            torques, (self.sea_hidden_state[:], self.sea_cell_state[:]) = self.network(
                self.sea_input, (self.sea_hidden_state, self.sea_cell_state)
            )

        # run network inference
        with torch.inference_mode():
            torques, (self.sea_hidden_state[:], self.sea_cell_state[:]) = self.network(
                self.sea_input, (self.sea_hidden_state, self.sea_cell_state)
            )
        self.computed_effort = torques.reshape(self._num_envs, self.num_joints)

        # clip the computed effort based on the motor limits
        self.applied_effort = self._clip_effort(self.computed_effort)

        # return torques
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


class ActuatorNetMLP(BaseActuatorNetMLP):
    """Extended Actuator model based on multi-layer perceptron and joint history."""

    def __init__(self, cfg, *args, **kwargs):
        # Delayed import to avoid circular dependency
        from .actuator_cfg import ActuatorNetMLPCfg

        self.cfg: ActuatorNetMLPCfg = cfg
        super().__init__(cfg, *args, **kwargs)

        history_length = max(self.cfg.input_idx) + 1

        if self.cfg.input_order == "pos_vel":
            self._joint_pos_error_history = torch.zeros(
                self._num_envs, history_length, self.num_joints, device=self._device
            )
            self._joint_vel_history = torch.zeros(
                self._num_envs, history_length, self.num_joints, device=self._device
            )
        elif self.cfg.input_order == "vel_pos":
            self._joint_vel_error_history = torch.zeros(
                self._num_envs, history_length, self.num_joints, device=self._device
            )
            self._joint_sin_history = torch.zeros(
                self._num_envs, history_length, self.num_joints, device=self._device
            )
            self._joint_cos_history = torch.zeros(
                self._num_envs, history_length, self.num_joints, device=self._device
            )
        else:
            raise ValueError(
                f"Invalid input order for MLP actuator net: {self.cfg.input_order}. Must be 'pos_vel' or 'vel_pos'."
            )

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        if self.cfg.input_order == "pos_vel":
            self._joint_pos_error_history = self._joint_pos_error_history.roll(1, 1)
            self._joint_pos_error_history[:, 0] = control_action.joint_positions - joint_pos
            self._joint_vel_history = self._joint_vel_history.roll(1, 1)
            self._joint_vel_history[:, 0] = joint_vel

            pos_input = torch.cat(
                [self._joint_pos_error_history[:, i].unsqueeze(2) for i in self.cfg.input_idx],
                dim=2
            ).view(self._num_envs * self.num_joints, -1)
            vel_input = torch.cat(
                [self._joint_vel_history[:, i].unsqueeze(2) for i in self.cfg.input_idx],
                dim=2
            ).view(self._num_envs * self.num_joints, -1)

            network_input = torch.cat(
                [pos_input * self.cfg.pos_scale, vel_input * self.cfg.vel_scale],
                dim=1
            )
        elif self.cfg.input_order == "vel_pos":
            self._joint_vel_error_history = self._joint_vel_error_history.roll(1, 1)
            self._joint_sin_history = self._joint_sin_history.roll(1, 1)
            self._joint_cos_history = self._joint_cos_history.roll(1, 1)

            self._joint_vel_error_history[:, 0] = control_action.joint_velocities - joint_vel
            self._joint_sin_history[:, 0] = torch.sin(joint_pos)
            self._joint_cos_history[:, 0] = torch.cos(joint_pos)

            vel_error_input = torch.cat(
                [self._joint_vel_error_history[:, i].unsqueeze(2) for i in self.cfg.input_idx],
                dim=2
            ).view(self._num_envs * self.num_joints, -1)
            sin_input = torch.cat(
                [self._joint_sin_history[:, i].unsqueeze(2) for i in self.cfg.input_idx],
                dim=2
            ).view(self._num_envs * self.num_joints, -1)
            cos_input = torch.cat(
                [self._joint_cos_history[:, i].unsqueeze(2) for i in self.cfg.input_idx],
                dim=2
            ).view(self._num_envs * self.num_joints, -1)

            network_input = torch.cat(
                [vel_error_input * self.cfg.vel_scale, sin_input, cos_input],
                dim=1
            )
        else:
            raise ValueError(
                f"Invalid input order for MLP actuator net: {self.cfg.input_order}. Must be 'pos_vel' or 'vel_pos'."
            )

        self._joint_vel[:] = joint_vel

        with torch.inference_mode():
            torques = self.network(network_input).view(self._num_envs, self.num_joints)
        self.computed_effort = torques.view(self._num_envs, self.num_joints) * self.cfg.torque_scale

        self.applied_effort = self._clip_effort(self.computed_effort)

        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action
    

class ActuatorNetKAN(DCMotor):
    """Extended Actuator model based on multi-layer perceptron and joint history."""

    def __init__(self, cfg, *args, **kwargs):
        # Delayed import to avoid circular dependency
        from .actuator_cfg import ActuatorNetKANCfg

        self.cfg: ActuatorNetKANCfg = cfg

        formula_str = read_file(self.cfg.symbolic_formula).getvalue().decode().strip()
        self.kan_symbolic_formula, formula_input_dim = self.parse_formula_to_lambda(formula_str)

        super().__init__(cfg, *args, **kwargs)
        
        history_length = max(self.cfg.input_idx) + 1

        if self.cfg.input_order == "pos_vel":
            self._joint_pos_error_history = torch.zeros(
                self._num_envs, history_length, self.num_joints, device=self._device
            )
            self._joint_vel_history = torch.zeros(
                self._num_envs, history_length, self.num_joints, device=self._device
            )
            input_dim = len(self.cfg.input_idx) * 2
        elif self.cfg.input_order == "vel_pos":
            self._joint_vel_error_history = torch.zeros(
                self._num_envs, history_length, self.num_joints, device=self._device
            )
            self._joint_sin_history = torch.zeros(
                self._num_envs, history_length, self.num_joints, device=self._device
            )
            self._joint_cos_history = torch.zeros(
                self._num_envs, history_length, self.num_joints, device=self._device
            )
            input_dim = len(self.cfg.input_idx) * 3
        else:
            raise ValueError(
                f"Invalid input order for MLP actuator net: {self.cfg.input_order}. Must be 'pos_vel' or 'vel_pos'."
            )

        assert input_dim >= formula_input_dim, (
            f"Symbolic formula requires at least {formula_input_dim} input dimensions, "
            f"but only {input_dim} are constructed from cfg.input_idx={self.cfg.input_idx}."
        )

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        if self.cfg.input_order == "pos_vel":
            self._joint_pos_error_history = self._joint_pos_error_history.roll(1, 1)
            self._joint_pos_error_history[:, 0] = control_action.joint_positions - joint_pos
            self._joint_vel_history = self._joint_vel_history.roll(1, 1)
            self._joint_vel_history[:, 0] = joint_vel

            pos_input = torch.cat(
                [self._joint_pos_error_history[:, i].unsqueeze(2) for i in self.cfg.input_idx],
                dim=2
            ).view(self._num_envs * self.num_joints, -1)
            vel_input = torch.cat(
                [self._joint_vel_history[:, i].unsqueeze(2) for i in self.cfg.input_idx],
                dim=2
            ).view(self._num_envs * self.num_joints, -1)

            x_input = torch.cat(
                [pos_input * self.cfg.pos_scale, vel_input * self.cfg.vel_scale],
                dim=1
            )
        elif self.cfg.input_order == "vel_pos":
            self._joint_vel_error_history = self._joint_vel_error_history.roll(1, 1)
            self._joint_sin_history = self._joint_sin_history.roll(1, 1)
            self._joint_cos_history = self._joint_cos_history.roll(1, 1)

            self._joint_vel_error_history[:, 0] = control_action.joint_velocities - joint_vel
            self._joint_sin_history[:, 0] = torch.sin(joint_pos)
            self._joint_cos_history[:, 0] = torch.cos(joint_pos)

            vel_error_input = torch.cat(
                [self._joint_vel_error_history[:, i].unsqueeze(2) for i in self.cfg.input_idx],
                dim=2
            ).view(self._num_envs * self.num_joints, -1)
            sin_input = torch.cat(
                [self._joint_sin_history[:, i].unsqueeze(2) for i in self.cfg.input_idx],
                dim=2
            ).view(self._num_envs * self.num_joints, -1)
            cos_input = torch.cat(
                [self._joint_cos_history[:, i].unsqueeze(2) for i in self.cfg.input_idx],
                dim=2
            ).view(self._num_envs * self.num_joints, -1)

            x_input = torch.cat(
                [vel_error_input * self.cfg.vel_scale, sin_input, cos_input],
                dim=1
            )
        else:
            raise ValueError(
                f"Invalid input order for MLP actuator net: {self.cfg.input_order}. Must be 'pos_vel' or 'vel_pos'."
            )

        self._joint_vel[:] = joint_vel

        with torch.inference_mode():
            torques = self.kan_symbolic_formula(x_input).view(self._num_envs, self.num_joints)
        self.computed_effort = torques.view(self._num_envs, self.num_joints) * self.cfg.torque_scale

        self.applied_effort = self._clip_effort(self.computed_effort)

        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action

    def parse_formula_to_lambda(self, formula_str: str):
        import re
        allowed_names = {
            'torch': torch,
            'sin': torch.sin,
            'cos': torch.cos,
            'tan': torch.tan,
            'exp': torch.exp,
            'log': torch.log,
            'sqrt': torch.sqrt,
            'abs': torch.abs,
            'pi': torch.pi,
        }
        print("[KAN] Original formula string:")
        print(formula_str)

        x_indices = [int(i) for i in re.findall(r'x_(\d+)', formula_str)]
        required_input_dim = max(x_indices) if x_indices else 0  # 그대로 사용

        for i in x_indices:
            formula_str = formula_str.replace(f'x_{i}', f'x[:, {i - 1}]')

        formula_func = eval(f'lambda x: {formula_str}', allowed_names)

        print(f"[KAN] Required input dim: {required_input_dim}")
        print("[KAN] Transformed PyTorch-compatible formula:")
        print(f"lambda x: {formula_str}")

        return formula_func, required_input_dim
    
class GearTransmissionPDActuator(DelayedPDActuator):
    """기어비를 명령/한계/관성에 일관 반영하는 DelayedPD 파생 액추에이터."""

    def __init__(self, cfg, *args, **kwargs):
        from .actuator_cfg import GearTransmissionPDCfg
        cfg: GearTransmissionPDCfg
        super().__init__(cfg, *args, **kwargs)

        # ----- 기어비 -----
        self._gamma = float(getattr(cfg, "gear_ratio", 1.0))

        # ----- 모터측 한계 추정 (입력 우선) -----
        # effort_limit / velocity_limit는 부모에서 tensor 또는 float로 들어올 수 있음 → 텐서화
        eff_lim_tensor = torch.as_tensor(self.effort_limit, device=self._device)
        vel_lim_tensor = torch.as_tensor(self.velocity_limit, device=self._device)

        if getattr(cfg, "motor_effort_limit", None) is not None:
            self._tau_m_max = float(cfg.motor_effort_limit)
        else:
            # 현재 effort_limit가 "조인트측"으로 세팅되었다고 가정 → 모터측으로 환원
            self._tau_m_max = (eff_lim_tensor / self._gamma).item() if torch.isfinite(eff_lim_tensor).all() else float("inf")

        if getattr(cfg, "motor_velocity_limit", None) is not None:
            self._omega_m_max = float(cfg.motor_velocity_limit)
        else:
            # 현재 velocity_limit가 "조인트측"으로 세팅되었다고 가정 → 모터측으로 환원
            self._omega_m_max = (vel_lim_tensor * self._gamma).item() if torch.isfinite(vel_lim_tensor).all() else float("inf")

        # ----- 조인트측 한계 재설정 (부모의 클리핑/검증은 조인트측 기준으로 동작) -----
        self.effort_limit = torch.as_tensor(self._tau_m_max * self._gamma, device=self._device)
        self.velocity_limit = torch.as_tensor(self._omega_m_max / self._gamma, device=self._device)

        # ----- 반사 관성 근사: armature *= γ^2 -----
        if bool(getattr(cfg, "scale_armature", True)):
            self.armature = torch.as_tensor(self.armature, device=self._device) #* (self._gamma ** 2)

        # ----- rate-limit 옵션 -----
        self._rate_limit = bool(getattr(cfg, "rate_limit", True))
        self._q_des_prev = None  # env reset마다 초기화

    # ------------------------------------------------
    # Utilities
    # ------------------------------------------------
    def _apply_gear_and_rate_limit(
        self,
        q_des_m: torch.Tensor | None,
        dq_des_m: torch.Tensor | None,
        dt: float,
    ):
        """모터측 명령 → (1/γ) → 조인트측 명령, 그리고 조인트측 속도 한계로 rate-limit."""
        if q_des_m is None:
            # (포지션 명령이 없으면 패스)
            return None, dq_des_m

        # 디바이스/형 안정화
        q_des_m = torch.as_tensor(q_des_m, device=self._device)
        dq_des_m = None if dq_des_m is None else torch.as_tensor(dq_des_m, device=self._device)

        # 모터 → 조인트 변환
        q_des_j = q_des_m * self._gamma
        dq_des_j = (dq_des_m * self._gamma) if dq_des_m is not None else torch.zeros_like(q_des_j, device=self._device)

        # rate-limit (조인트측 최대속도 = ω_m,max / γ)
        if self._rate_limit:
            vlim = torch.as_tensor(self.velocity_limit, device=self._device)  # 조인트측
            if torch.isfinite(vlim).all():
                dq_step_max = vlim * dt  # 브로드캐스팅 허용
                if self._q_des_prev is None:
                    self._q_des_prev = q_des_j.clone()
                dq_step = torch.clamp(q_des_j - self._q_des_prev, min=-dq_step_max, max=dq_step_max)
                q_des_j = self._q_des_prev + dq_step
                self._q_des_prev = q_des_j.clone()

        return q_des_j, dq_des_j

    # ------------------------------------------------
    # Lifecycle
    # ------------------------------------------------
    def reset(self, env_ids):
        """부모 초기화 + 내부 상태 초기화."""
        super().reset(env_ids)
        # 부분 리셋 시에도 안전하게 초기화
        self._q_des_prev = None

    # ------------------------------------------------
    # Main step
    # ------------------------------------------------
    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        """상위 명령(모터측으로 해석)을 조인트측으로 변환 → DelayedPDActuator.compute."""
        # 물리 dt (부모에서 보통 보관), 없으면 보수값
        dt = float(getattr(self, "_physics_dt", 1.0 / 60.0))

        # 상위 명령 가져오기
        q_des_m = control_action.joint_positions
        dq_des_m = control_action.joint_velocities

        # 기어 전송 + rate-limit
        q_des_j, dq_des_j = self._apply_gear_and_rate_limit(q_des_m, dq_des_m, dt)

        # 변환된 명령을 control_action에 반영 (None일 경우 기존값 유지)
        if q_des_j is not None:
            control_action.joint_positions = q_des_j
            control_action.joint_velocities = dq_des_j

        # 부모 로직(지연→PD→클리핑)
        control_action = super().compute(control_action, joint_pos, joint_vel)
        control_action.joint_efforts = control_action.joint_efforts * self._gamma 
        return control_action
