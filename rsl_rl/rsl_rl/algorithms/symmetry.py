"""
左右对称状态增强模块 —— 适配 IsaacGym Go2 / LeggedRobot
基于 legged_robot_config.py 中的 obs 布局精确实现

obs_buf (n_proprio=53) 布局（来自 compute_observations）：
  [0:3]   base_ang_vel * scale          (wx, wy, wz)
  [3:5]   imu_obs (roll, pitch)
  [5:6]   commands[:,0] vx
  [6:7]   commands[:,1] vy
  [7:8]   commands[:,2] wz_cmd (delta_yaw)
  [8:11]  0 * commands (zero padding, 3维)
  [11:12] (env_class != 17).float()     parkour flag
  [12:13] (env_class == 17).float()     flat flag
  [13:25] reindex(dof_pos - default)    12维
  [25:37] reindex(dof_vel)              12维
  [37:49] reindex(action_history[-1])   12维
  [49:53] reindex_feet(contact - 0.5)   4维 [FR, FL, RR, RL]

reindex = [3,4,5, 0,1,2, 9,10,11, 6,7,8]
原始URDF关节顺序（假设）：[FL_hip,FL_thigh,FL_calf, FR_hip,FR_thigh,FR_calf,
                           RL_hip,RL_thigh,RL_calf, RR_hip,RR_thigh,RR_calf]
reindex后：               [FR_hip,FR_thigh,FR_calf, FL_hip,FL_thigh,FL_calf,
                           RR_hip,RR_thigh,RR_calf, RL_hip,RL_thigh,RL_calf]

完整obs结构：
  [0 : n_rhythm]                           phase_obs (sin/cos, 不变)
  [n_rhythm : n_rhythm+n_proprio]          当前帧 proprio
  [n_rhythm+n_proprio : +n_scan]           scan heights (不变，或跳过)
  [...  : +n_priv]                         priv states (lin_vel需处理)
  [...  : +n_priv_latent]                  priv latent (不变)
  [...  : +history_len*n_proprio]          history buf (每帧proprio需变换)
"""

from __future__ import annotations
import torch

__all__ = ["SymmetricAugmentation"]

# ================================================================
# reindex后12维关节顺序：
#   0  FR_hip    1  FR_thigh   2  FR_calf
#   3  FL_hip    4  FL_thigh   5  FL_calf
#   6  RR_hip    7  RR_thigh   8  RR_calf
#   9  RL_hip   10  RL_thigh  11  RL_calf
#
# 左右对称：FR(0,1,2)<->FL(3,4,5)，RR(6,7,8)<->RL(9,10,11)
# hip abduction(索引0,3,6,9)符号取反（Go2左右hip默认角度符号相反）
# contact 4维：[FR=0, FL=1, RR=2, RL=3]
# ================================================================

_FR = [0, 1, 2]
_FL = [3, 4, 5]
_RR = [6, 7, 8]
_RL = [9, 10, 11]
_HIP = [0, 3, 6, 9]  # hip abduction，左右符号相反


def _swap_legs_12(x: torch.Tensor) -> torch.Tensor:
    """交换12维关节数据左右腿并对hip取反。shape: (..., 12)"""
    out = x.clone()
    out[..., _FR], out[..., _FL] = x[..., _FL].clone(), x[..., _FR].clone()
    out[..., _RR], out[..., _RL] = x[..., _RL].clone(), x[..., _RR].clone()
    out[..., _HIP] *= -1.0
    return out


def _swap_contact_4(x: torch.Tensor) -> torch.Tensor:
    """交换4维接触 [FR,FL,RR,RL] 左右。shape: (..., 4)"""
    out = x.clone()
    out[..., 0], out[..., 1] = x[..., 1].clone(), x[..., 0].clone()  # FR<->FL
    out[..., 2], out[..., 3] = x[..., 3].clone(), x[..., 2].clone()  # RR<->RL
    return out


def _mirror_proprio(p: torch.Tensor) -> torch.Tensor:
    """
    对单帧53维proprio做左右镜像。
    p shape: (N, 53)
    """
    f = p.clone()
    d = f.device

    # [0:3] ang_vel: wx取反(roll rate), wy不变, wz取反(yaw rate)
    f[:, 0:3] *= torch.tensor([-1., 1., -1.], device=d)

    # [3:5] imu: roll取反, pitch不变
    f[:, 3] *= -1.

    # [5:8] commands: vx不变, vy取反, wz_cmd(delta_yaw)取反
    f[:, 5:8] *= torch.tensor([1., -1., -1.], device=d)

    # [8:13] zero_padding + env_flags：与左右无关，不变

    # [13:25] joint_pos
    f[:, 13:25] = _swap_legs_12(f[:, 13:25])

    # [25:37] joint_vel
    f[:, 25:37] = _swap_legs_12(f[:, 25:37])

    # [37:49] action
    f[:, 37:49] = _swap_legs_12(f[:, 37:49])

    # [49:53] contact [FR, FL, RR, RL]
    f[:, 49:53] = _swap_contact_4(f[:, 49:53])

    return f


def _mirror_priv(priv: torch.Tensor, n_priv: int = 9) -> torch.Tensor:
    """
    对priv_states做镜像。
    n_priv=9: [lin_vel(3), zeros(3), zeros(3)]
    lin_vel: vy取反(lateral)，vx/vz不变
    """
    f = priv.clone()
    if n_priv >= 2:
        f[:, 1] *= -1.  # lin_vel_y 取反
    return f


def _mirror_full_obs(
    obs: torch.Tensor,
    n_rhythm: int,
    n_proprio: int,
    n_scan: int,
    n_priv: int,
    n_priv_latent: int,
    history_len: int,
    mirror_priv: bool = False,
) -> torch.Tensor:
    """
    对完整obs做左右镜像变换。
    obs layout:
      [0          : n_rhythm]                  phase_obs
      [n_rhythm   : n_rhythm+n_proprio]        current proprio
      [+n_scan    : +n_scan]                   scan (不变)
      [+n_priv    : +n_priv]                   priv states
      [+n_priv_latent : ...]                   priv latent (不变)
      [末尾 history_len*n_proprio]              history
    """
    obs = obs.clone()

    # 1. 当前帧proprio
    s = n_rhythm
    e = s + n_proprio
    obs[:, s:e] = _mirror_proprio(obs[:, s:e])

    # 2. priv states（可选，actor obs通常不含priv真值）
    if mirror_priv and n_priv > 0:
        priv_s = n_rhythm + n_proprio + n_scan
        priv_e = priv_s + n_priv
        if priv_e <= obs.shape[1]:
            obs[:, priv_s:priv_e] = _mirror_priv(obs[:, priv_s:priv_e], n_priv)

    # 3. history buf（末尾 history_len * n_proprio 维）
    if history_len > 0:
        hist_s = obs.shape[1] - history_len * n_proprio
        for i in range(history_len):
            hs = hist_s + i * n_proprio
            he = hs + n_proprio
            obs[:, hs:he] = _mirror_proprio(obs[:, hs:he])

    return obs


class SymmetricAugmentation:
    """
    左右对称数据增强，将batch扩充一倍。
    支持 actor obs 和 critic obs（含priv）分别处理。

    使用方式：
        aug = SymmetricAugmentation(cfg, enabled=True)
        obs_aug, critic_obs_aug, acts_aug, rews_aug, dones_aug = aug.augment(
            obs, critic_obs, actions, rewards, dones
        )
    """

    def __init__(self, env_cfg, enabled: bool = True):
        self.enabled = enabled
        cfg = env_cfg

        self.n_rhythm      = getattr(cfg.env, 'n_rhythm',      4)
        self.n_proprio     = getattr(cfg.env, 'n_proprio',     53)
        self.n_scan        = getattr(cfg.env, 'n_scan',        132)
        self.n_priv        = getattr(cfg.env, 'n_priv',        9)
        self.n_priv_latent = getattr(cfg.env, 'n_priv_latent', 29)
        self.history_len   = getattr(cfg.env, 'history_len',   10)

    @torch.no_grad()
    def augment(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor | None,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ):
        """
        将一个rollout step的数据扩充一倍（原始 + 左右镜像）。

        Returns:
            obs_aug, critic_obs_aug, actions_aug, rewards_aug, dones_aug
        """
        if not self.enabled:
            return obs, critic_obs, actions, rewards, dones

        # mirror actor obs
        obs_mirror = _mirror_full_obs(
            obs,
            n_rhythm=self.n_rhythm,
            n_proprio=self.n_proprio,
            n_scan=self.n_scan,
            n_priv=self.n_priv,
            n_priv_latent=self.n_priv_latent,
            history_len=self.history_len,
            mirror_priv=False,  # actor obs 不含真实priv
        )
        obs_aug = torch.cat([obs, obs_mirror], dim=0)

        # mirror critic obs（含priv states）
        if critic_obs is not None:
            critic_mirror = _mirror_full_obs(
                critic_obs,
                n_rhythm=self.n_rhythm,
                n_proprio=self.n_proprio,
                n_scan=self.n_scan,
                n_priv=self.n_priv,
                n_priv_latent=self.n_priv_latent,
                history_len=self.history_len,
                mirror_priv=True,  # critic obs 含真实priv，需要处理lin_vel
            )
            critic_obs_aug = torch.cat([critic_obs, critic_mirror], dim=0)
        else:
            critic_obs_aug = None

        # mirror actions
        actions_mirror = _swap_legs_12(actions)
        actions_aug = torch.cat([actions, actions_mirror], dim=0)

        # rewards/dones 直接复制（对称不改变奖励）
        rewards_aug = rewards.repeat(2, *([1] * (rewards.dim() - 1)))
        dones_aug   = dones.repeat(2, *([1] * (dones.dim() - 1)))

        return obs_aug, critic_obs_aug, actions_aug, rewards_aug, dones_aug

    @torch.no_grad()
    def verify_symmetry(self, obs: torch.Tensor, tol: float = 1e-5) -> bool:
        """验证：对obs做两次镜像应还原原始值（对合性检验）"""
        obs_m1 = _mirror_full_obs(
            obs, self.n_rhythm, self.n_proprio, self.n_scan,
            self.n_priv, self.n_priv_latent, self.history_len
        )
        obs_m2 = _mirror_full_obs(
            obs_m1, self.n_rhythm, self.n_proprio, self.n_scan,
            self.n_priv, self.n_priv_latent, self.history_len
        )
        ok = torch.allclose(obs, obs_m2, atol=tol)
        if not ok:
            diff = (obs - obs_m2).abs().max().item()
            print(f"[SymAug] 对合性检验失败，最大误差: {diff:.2e}")
        else:
            print("[SymAug] 对合性检验通过 ✓")
        return ok