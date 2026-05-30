import os

import torch
import torch.nn as nn

from rsl_rl.modules.depth_backbone import DepthOnlyFCBackbone58x87, RecurrentDepthBackbone


def latest_model_path(logdir, suffix="base_jit.pt"):
    candidates = [
        os.path.join(logdir, name)
        for name in os.listdir(logdir)
        if name.endswith(suffix)
    ]
    if not candidates:
        traced = os.path.join(logdir, "traced")
        if os.path.isdir(traced):
            candidates = [
                os.path.join(traced, name)
                for name in os.listdir(traced)
                if name.endswith(suffix)
            ]
    if not candidates:
        raise FileNotFoundError(f"No *{suffix} found in {logdir} or {os.path.join(logdir, 'traced')}")
    return max(candidates, key=_checkpoint_sort_key)


def _checkpoint_sort_key(path):
    stem = os.path.basename(path).replace(".pt", "")
    nums = []
    for part in stem.replace("_", "-").split("-"):
        if part.isdigit():
            nums.append(int(part))
    return nums[-1] if nums else -1


class ConfigView:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                value = ConfigView(value)
            setattr(self, key, value)


class ExportedHardwareVisionPolicy(nn.Module):
    def __init__(self, cfg, base_model):
        super().__init__()
        env_cfg = ConfigView(cfg)
        policy_cfg = cfg.get("policy", {})

        self.n_proprio = cfg["env"]["n_proprio"]
        self.n_scan = cfg["env"]["n_scan"]
        self.n_priv = cfg["env"]["n_priv"]
        self.n_priv_latent = cfg["env"]["n_priv_latent"]
        self.history_len = cfg["env"]["history_len"]
        self.depth_update_interval = cfg["depth"].get("update_interval", 5)
        self.base_obs_dim = (
            self.n_proprio
            + self.n_scan
            + self.n_priv
            + self.n_priv_latent
            + self.history_len * self.n_proprio
        )

        scan_encoder_dims = policy_cfg.get("scan_encoder_dims", [128, 64, 32])

        self.base_model = base_model
        depth_backbone = DepthOnlyFCBackbone58x87(
            self.n_proprio,
            scan_encoder_dims[-1],
            cfg.get("depth_encoder", {}).get("hidden_dims", 512),
        )
        self.depth_encoder = RecurrentDepthBackbone(depth_backbone, env_cfg)
        self._last_depth_latent_yaw = None

    def load_vision_state(self, state):
        self.depth_encoder.load_state_dict(state["depth_encoder_state_dict"], strict=True)

    def reset(self):
        self.depth_encoder.reset()
        self._last_depth_latent_yaw = None

    @torch.no_grad()
    def forward(self, proprio, proprio_history, depth_image, update_depth=True):
        if update_depth or self._last_depth_latent_yaw is None:
            obs_prop_depth = proprio.clone()
            obs_prop_depth[:, 6:8] = 0.
            self._last_depth_latent_yaw = self.depth_encoder(depth_image, obs_prop_depth)

        depth_latent = self._last_depth_latent_yaw[:, :-2]
        self.last_predicted_yaw = self._last_depth_latent_yaw[:, -2:] * 1.5

        base_obs = torch.zeros(
            proprio.shape[0],
            self.base_obs_dim,
            device=proprio.device,
            dtype=proprio.dtype,
        )
        actor_proprio = proprio.clone()
        base_obs[:, :self.n_proprio] = actor_proprio
        history_start = self.n_proprio + self.n_scan + self.n_priv + self.n_priv_latent
        base_obs[:, history_start:] = proprio_history.reshape(proprio.shape[0], -1)

        return self.base_model(base_obs, depth_latent)


def load_hardware_vision_policy(logdir, cfg, device):
    base_model_path = latest_model_path(logdir, "base_jit.pt")
    vision_model_path = latest_model_path(logdir, "vision_weight.pt")

    base_model = torch.jit.load(base_model_path, map_location=device)
    base_model.eval()

    model = ExportedHardwareVisionPolicy(cfg, base_model).to(device)
    vision_state = torch.load(vision_model_path, map_location=device)
    model.load_vision_state(vision_state)
    model.eval()
    return model, f"base={base_model_path}, vision={vision_model_path}"
