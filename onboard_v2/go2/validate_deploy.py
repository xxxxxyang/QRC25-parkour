#!/usr/bin/env python3

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

def latest_model_path(logdir: Path, suffix: str):
    candidates = [p for p in logdir.iterdir() if p.is_file() and p.name.endswith(suffix)]
    traced = logdir / "traced"
    if not candidates and traced.is_dir():
        candidates = [p for p in traced.iterdir() if p.is_file() and p.name.endswith(suffix)]
    if not candidates:
        raise FileNotFoundError(f"No *{suffix} found in {logdir} or {traced}")

    def checkpoint_key(path: Path):
        stem = path.name.replace(".pt", "")
        nums = []
        for part in stem.replace("_", "-").split("-"):
            if part.isdigit():
                nums.append(int(part))
        return nums[-1] if nums else -1

    return max(candidates, key=checkpoint_key)


def run_cmd(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"<failed: {exc}>"


def derive_expected(cfg):
    env = cfg["env"]
    depth = cfg["depth"]
    return {
        "n_proprio": env["n_proprio"],
        "n_scan": env["n_scan"],
        "n_priv": env["n_priv"],
        "n_priv_latent": env["n_priv_latent"],
        "history_len": env["history_len"],
        "num_actions": env["num_actions"],
        "base_obs_dim": (
            env["n_proprio"]
            + env["n_scan"]
            + env["n_priv"]
            + env["n_priv_latent"]
            + env["history_len"] * env["n_proprio"]
        ),
        "depth_resized": depth["resized"],
        "depth_update_interval": depth.get("update_interval", 5),
    }


def check_config(cfg):
    failures = []
    env = cfg.get("env", {})
    depth = cfg.get("depth", {})
    required_env = ["n_proprio", "n_scan", "n_priv", "n_priv_latent", "history_len", "num_actions"]
    required_depth = ["resized", "original"]
    for key in required_env:
        if key not in env:
            failures.append(f"missing env.{key}")
    for key in required_depth:
        if key not in depth:
            failures.append(f"missing depth.{key}")
    if failures:
        return failures
    if env["n_proprio"] <= 0 or env["history_len"] <= 0:
        failures.append("env dimensions must be positive")
    if depth["resized"][0] <= 0 or depth["resized"][1] <= 0:
        failures.append("depth.resized must be positive")
    return failures


def check_so():
    machine = platform.machine().lower()
    arch_dir = "aarch64" if machine == "aarch64" else "x86"
    base_dir = Path(__file__).resolve().parent
    so_path = base_dir / arch_dir / "crc_module.so"
    alt_path = base_dir / ("x86" if arch_dir == "aarch64" else "aarch64") / "crc_module.so"
    print(f"[so] host={machine}, selected={so_path}")
    if not so_path.exists():
        print(f"[so] missing selected .so, alternate exists={alt_path.exists()}")
        return False

    file_out = run_cmd(["file", str(so_path)])
    print(f"[so] file: {file_out}")
    nm_out = run_cmd(["nm", "-D", str(so_path)])
    symbols = [line for line in nm_out.splitlines() if "PyInit_crc_module" in line]
    strings_out = run_cmd(["strings", str(so_path)])
    has_get_crc = any(line == "get_crc" for line in strings_out.splitlines())
    print(f"[so] exported={symbols or '<none>'}, has_get_crc_string={has_get_crc}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        required=True,
        type=Path,
        help="Path to the exported traced directory or its parent run directory",
    )
    args = parser.parse_args()

    logdir = args.logdir.resolve()
    if (logdir / "config.json").is_file():
        traced_dir = logdir
    elif (logdir / "traced" / "config.json").is_file():
        traced_dir = logdir / "traced"
    else:
        raise FileNotFoundError(f"config.json not found under {logdir}")

    with open(traced_dir / "config.json", "r") as f:
        cfg = json.load(f)

    print(f"[cfg] traced_dir={traced_dir}")
    print(f"[cfg] env={ {k: cfg['env'][k] for k in ['n_proprio','n_scan','n_priv','n_priv_latent','history_len','num_actions']} }")
    print(f"[cfg] depth={ {k: cfg['depth'][k] for k in ['original','resized','update_interval','near_clip','far_clip']} }")
    expected = derive_expected(cfg)
    print(f"[cfg] derived_base_obs_dim={expected['base_obs_dim']}")

    failures = check_config(cfg)
    if failures:
        print("[cfg] mismatches:")
        for item in failures:
            print(f"  - {item}")
    else:
        print("[cfg] dimension checks passed")

    base_path = latest_model_path(traced_dir, "base_jit.pt")
    vision_path = latest_model_path(traced_dir, "vision_weight.pt")
    print(f"[model] base={base_path}")
    print(f"[model] vision={vision_path}")

    check_so()

    try:
        import torch
    except Exception as exc:
        print(f"[torch] unavailable: {exc}")
        print("[torch] skip runtime forward check")
        sys.exit(1 if failures else 0)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from vision_policy import load_hardware_vision_policy

    device = "cpu"
    model, loaded_paths = load_hardware_vision_policy(str(traced_dir), cfg, device)
    print(f"[torch] loaded {loaded_paths}")
    print(f"[torch] base_obs_dim={model.base_obs_dim}, depth_update_interval={model.depth_update_interval}")
    if model.base_obs_dim != expected["base_obs_dim"]:
        failures.append(
            f"base_obs_dim mismatch: model wrapper {model.base_obs_dim}, config-derived {expected['base_obs_dim']}"
        )
    if model.depth_update_interval != expected["depth_update_interval"]:
        failures.append(
            "depth_update_interval mismatch: "
            f"model wrapper {model.depth_update_interval}, config-derived {expected['depth_update_interval']}"
        )

    proprio = torch.zeros(1, cfg["env"]["n_proprio"], device=device)
    proprio_hist = torch.zeros(1, cfg["env"]["history_len"], cfg["env"]["n_proprio"], device=device)
    resized = cfg["depth"]["resized"]
    depth = torch.zeros(1, resized[1], resized[0], device=device)

    with torch.inference_mode():
        action = model(proprio, proprio_hist, depth, update_depth=True)
        cached_action = model(proprio, proprio_hist, depth, update_depth=False)

    print(
        f"[torch] action_shape={tuple(action.shape)} "
        f"cached_shape={tuple(cached_action.shape)} "
        f"finite={bool(torch.isfinite(action).all() and torch.isfinite(cached_action).all())} "
        f"range=({float(action.min()):.4f}, {float(action.max()):.4f})"
    )

    if action.shape != (1, cfg["env"]["num_actions"]):
        failures.append(f"action shape mismatch: {tuple(action.shape)}")
    if cached_action.shape != action.shape:
        failures.append("cached action shape mismatch")
    if not torch.isfinite(action).all() or not torch.isfinite(cached_action).all():
        failures.append("non-finite model output")

    if failures:
        print("[result] FAILED")
        for item in failures:
            print(f"  - {item}")
        sys.exit(1)

    print("[result] OK")


if __name__ == "__main__":
    main()
