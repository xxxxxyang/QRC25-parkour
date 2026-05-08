# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Capture a short simulated motion sequence and compose selected frames into
# one image for result visualization.

from legged_gym import LEGGED_GYM_ROOT_DIR

import argparse
import faulthandler
import math
import os
from datetime import datetime

import isaacgym  # noqa: F401
from isaacgym import gymapi  # noqa: F401
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry
import cv2
import numpy as np
import torch
from tqdm import tqdm


TERRAIN_KEYS = [
    "smooth slope",
    "rough slope up",
    "rough slope down",
    "rough stairs up",
    "rough stairs down",
    "discrete",
    "stepping stones",
    "gaps",
    "smooth flat",
    "pit",
    "wall",
    "platform",
    "large stairs up",
    "large stairs down",
    "parkour",
    "parkour_hurdle",
    "parkour_flat",
    "parkour_step",
    "parkour_gap",
    "demo",
]


def _parse_vec3(value):
    parts = [float(x) for x in value.replace(",", " ").split()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected 3 numbers, e.g. '-1.8,2.2,1.0'")
    return parts


def _parse_steps(value):
    if value is None or value == "":
        return None
    return [int(x) for x in value.replace(",", " ").split()]


def parse_sequence_args(base_args=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--seq_start", type=int, default=120, help="First step to capture.")
    parser.add_argument("--seq_interval", type=int, default=25, help="Step gap between captured frames.")
    parser.add_argument("--seq_frames", type=int, default=6, help="Number of frames to capture.")
    parser.add_argument("--seq_steps", type=_parse_steps, default=None, help="Explicit capture steps, e.g. 120,145,170.")
    parser.add_argument("--seq_total_steps", type=int, default=None, help="Total simulation steps. Defaults to last capture step.")
    parser.add_argument("--seq_env_id", type=int, default=0, help="Environment id to follow and capture.")
    parser.add_argument("--seq_output", type=str, default=None, help="Output image path.")
    parser.add_argument("--seq_layout", choices=["strip", "grid"], default="strip", help="Final image layout.")
    parser.add_argument("--seq_grid_cols", type=int, default=0, help="Columns for grid layout. 0 means automatic.")
    parser.add_argument("--seq_frame_width", type=int, default=640, help="Resize each captured frame to this width. 0 disables resizing.")
    parser.add_argument("--seq_gap", type=int, default=12, help="Pixels between frames in the final image.")
    parser.add_argument("--seq_no_label", dest="seq_label", action="store_false", default=True, help="Do not draw time labels.")
    parser.add_argument("--seq_capture_mode", choices=["camera", "viewer"], default="camera", help="Capture with an offscreen camera sensor or the Isaac Gym viewer.")
    parser.add_argument("--seq_camera_width", type=int, default=1280, help="Offscreen camera image width.")
    parser.add_argument("--seq_camera_height", type=int, default=720, help="Offscreen camera image height.")
    parser.add_argument("--seq_camera_fov", type=float, default=70.0, help="Offscreen camera horizontal FOV in degrees.")
    parser.add_argument("--seq_fixed_camera", action="store_true", default=False, help="Keep camera fixed at the initial pose.")
    parser.add_argument("--seq_camera_offset", type=_parse_vec3, default=[-1.8, 2.2, 1.0], help="Camera position relative to robot.")
    parser.add_argument("--seq_camera_lookat_offset", type=_parse_vec3, default=[0.0, 0.0, 0.25], help="Fixed-camera look-at offset.")
    parser.add_argument("--seq_terrain", choices=["gap", "flat", "hurdle", "step", "mixed", "config"], default="gap")
    parser.add_argument("--seq_debug_viz", action="store_true", default=False, help="Keep debug lines in viewer screenshots.")

    ns, unknown = parser.parse_known_args()

    import sys

    sys.argv = [sys.argv[0]] + unknown
    if base_args is not None:
        for key, value in vars(ns).items():
            setattr(base_args, key, value)
        return base_args
    return ns


def build_capture_steps(args):
    if args.seq_steps is not None:
        steps = args.seq_steps
    else:
        steps = [args.seq_start + i * args.seq_interval for i in range(args.seq_frames)]
    steps = sorted(set(steps))
    if not steps or steps[0] < 0:
        raise ValueError("capture steps must be non-empty and non-negative")
    return steps


def make_terrain_dict(preset):
    terrain = {key: 0.0 for key in TERRAIN_KEYS}
    if preset == "config":
        return None
    if preset == "gap":
        terrain["parkour_gap"] = 1.0
    elif preset == "flat":
        terrain["parkour_flat"] = 1.0
    elif preset == "hurdle":
        terrain["parkour_hurdle"] = 1.0
    elif preset == "step":
        terrain["parkour_step"] = 1.0
    elif preset == "mixed":
        terrain["parkour_flat"] = 0.25
        terrain["parkour_hurdle"] = 0.25
        terrain["parkour_step"] = 0.25
        terrain["parkour_gap"] = 0.25
    else:
        raise ValueError(f"Unknown terrain preset: {preset}")
    return terrain


def apply_visualization_cfg(env_cfg, args):
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0

    if args.num_envs is None:
        env_cfg.env.num_envs = max(args.seq_env_id + 1, min(env_cfg.env.num_envs, 8))
    else:
        args.num_envs = max(args.num_envs, args.seq_env_id + 1)

    env_cfg.env.episode_length_s = 60
    env_cfg.commands.resampling_time = 60
    env_cfg.commands.curriculum = False

    if args.rows is None:
        env_cfg.terrain.num_rows = 5
    if args.cols is None:
        env_cfg.terrain.num_cols = 5
    env_cfg.terrain.height = [0.02, 0.02]

    terrain_dict = make_terrain_dict(args.seq_terrain)
    if terrain_dict is not None:
        env_cfg.terrain.terrain_dict = terrain_dict
        env_cfg.terrain.terrain_proportions = list(terrain_dict.values())
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = False

    env_cfg.depth.angle = [0, 1]
    env_cfg.noise.add_noise = False
    # LeggedRobot.compute_observations() always reads friction_coeffs_tensor;
    # this tensor is created in _create_envs() only when randomize_friction is on.
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.env.sequence_camera = args.seq_capture_mode == "camera"


def default_output_path(log_dir):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(log_dir, "motion_sequence", f"motion_sequence_{stamp}.png")


def resolve_output_path(path, log_dir):
    if path is None:
        return default_output_path(log_dir)
    if os.path.isabs(path):
        return path
    return os.path.join(LEGGED_GYM_ROOT_DIR, path)


def get_sequence_root_pos(env, args):
    return env.root_states[args.seq_env_id, :3].detach().cpu().numpy()


def set_capture_camera(env, args, camera_anchor_pos=None):
    env.lookat_id = args.seq_env_id
    offset = torch.tensor(args.seq_camera_offset, dtype=torch.float, device=env.device)

    if args.seq_fixed_camera:
        env.free_cam = True
        root = get_sequence_root_pos(env, args) if camera_anchor_pos is None else camera_anchor_pos
        cam_pos = root + np.array(args.seq_camera_offset, dtype=np.float32)
        lookat = root + np.array(args.seq_camera_lookat_offset, dtype=np.float32)
        env.set_camera(cam_pos, lookat)
        return root.copy()
    else:
        env.free_cam = False
        env.lookat_vec = offset
        return None


def capture_viewer_frame(env, args, frame_path, camera_anchor_pos=None):
    if env.viewer is None:
        raise RuntimeError("Viewer is not available. Run without --headless to capture sequence images.")
    camera_anchor_pos = set_capture_camera(env, args, camera_anchor_pos)
    env.render(sync_frame_time=False)
    env.gym.write_viewer_image_to_file(env.viewer, frame_path)
    return camera_anchor_pos


def create_sequence_camera(env, args):
    camera_props = gymapi.CameraProperties()
    camera_props.width = args.seq_camera_width
    camera_props.height = args.seq_camera_height
    camera_props.horizontal_fov = args.seq_camera_fov
    camera_handle = env.gym.create_camera_sensor(env.envs[args.seq_env_id], camera_props)
    if camera_handle < 0:
        raise RuntimeError("Failed to create offscreen camera sensor.")
    return camera_handle


def set_sequence_camera_location(env, args, camera_handle, camera_anchor_pos=None):
    if args.seq_fixed_camera:
        root = get_sequence_root_pos(env, args) if camera_anchor_pos is None else camera_anchor_pos
    else:
        root = get_sequence_root_pos(env, args)

    cam_pos = root + np.array(args.seq_camera_offset, dtype=np.float32)
    lookat = root + np.array(args.seq_camera_lookat_offset, dtype=np.float32)
    env.gym.set_camera_location(
        camera_handle,
        env.envs[args.seq_env_id],
        gymapi.Vec3(*cam_pos),
        gymapi.Vec3(*lookat),
    )
    return root.copy() if args.seq_fixed_camera else None


def capture_camera_frame(env, args, camera_handle, frame_path, camera_anchor_pos=None):
    camera_anchor_pos = set_sequence_camera_location(env, args, camera_handle, camera_anchor_pos)
    if env.device != "cpu":
        env.gym.fetch_results(env.sim, True)
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    image = env.gym.get_camera_image(
        env.sim,
        env.envs[args.seq_env_id],
        camera_handle,
        gymapi.IMAGE_COLOR,
    )
    image = np.asarray(image, dtype=np.uint8).reshape(args.seq_camera_height, args.seq_camera_width, 4)
    frame = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(frame_path, frame):
        raise RuntimeError(f"Failed to write captured frame: {frame_path}")
    return camera_anchor_pos


def capture_frame(env, args, frame_path, camera_handle=None, camera_anchor_pos=None):
    if args.seq_capture_mode == "viewer":
        return capture_viewer_frame(env, args, frame_path, camera_anchor_pos)
    else:
        return capture_camera_frame(env, args, camera_handle, frame_path, camera_anchor_pos)


def draw_label(frame, text):
    labeled = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.55, labeled.shape[1] / 1200.0)
    thickness = max(1, int(round(labeled.shape[1] / 600.0)))
    pad = max(8, int(round(labeled.shape[1] / 90.0)))
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x1, y1 = pad, pad
    x2, y2 = x1 + tw + 2 * pad, y1 + th + baseline + 2 * pad

    overlay = labeled.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.68, labeled, 0.32, 0, labeled)
    cv2.putText(labeled, text, (x1 + pad, y1 + pad + th), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return labeled


def resize_frame(frame, width):
    if width is None or width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w == width:
        return frame
    height = int(round(h * width / w))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def load_frames(frame_infos, frame_width, label, dt):
    frames = []
    for path, step in frame_infos:
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Failed to read captured frame: {path}")
        frame = resize_frame(frame, frame_width)
        if label:
            frame = draw_label(frame, f"t={step * dt:.2f}s")
        frames.append(frame)
    return frames


def compose_strip(frames, gap):
    gap = max(0, gap)
    height = max(frame.shape[0] for frame in frames)
    width = sum(frame.shape[1] for frame in frames) + gap * (len(frames) - 1)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    x = 0
    for frame in frames:
        h, w = frame.shape[:2]
        y = (height - h) // 2
        canvas[y:y + h, x:x + w] = frame
        x += w + gap
    return canvas


def compose_grid(frames, gap, cols):
    gap = max(0, gap)
    if cols <= 0:
        cols = int(math.ceil(math.sqrt(len(frames))))
    rows = int(math.ceil(len(frames) / cols))
    cell_w = max(frame.shape[1] for frame in frames)
    cell_h = max(frame.shape[0] for frame in frames)
    width = cols * cell_w + gap * (cols - 1)
    height = rows * cell_h + gap * (rows - 1)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    for idx, frame in enumerate(frames):
        row, col = divmod(idx, cols)
        h, w = frame.shape[:2]
        x = col * (cell_w + gap) + (cell_w - w) // 2
        y = row * (cell_h + gap) + (cell_h - h) // 2
        canvas[y:y + h, x:x + w] = frame
    return canvas


def compose_sequence_image(frame_infos, output_path, args, dt):
    frames = load_frames(frame_infos, args.seq_frame_width, args.seq_label, dt)
    if args.seq_layout == "grid":
        canvas = compose_grid(frames, args.seq_gap, args.seq_grid_cols)
    else:
        canvas = compose_strip(frames, args.seq_gap)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not cv2.imwrite(output_path, canvas):
        raise RuntimeError(f"Failed to write sequence image: {output_path}")


def compute_depth_latent(env, ppo_runner, depth_encoder, infos, obs):
    if not env.cfg.depth.use_camera:
        return None
    if infos.get("depth") is None:
        return None

    obs_student = obs[:, :env.cfg.env.n_proprio].clone()
    if obs_student.shape[1] >= 8:
        obs_student[:, 6:8] = 0

    with torch.no_grad():
        depth_out = depth_encoder(infos["depth"].clone().to(ppo_runner.device), obs_student)

    if isinstance(depth_out, dict):
        for key in ("belief", "depth_latent", "latent"):
            if key in depth_out:
                return depth_out[key]
        return None

    if isinstance(depth_out, (tuple, list)):
        depth_out = depth_out[0]

    if torch.is_tensor(depth_out) and depth_out.ndim == 2 and depth_out.shape[1] > 2:
        yaw = depth_out[:, -2:]
        if obs.shape[1] >= 8:
            obs[:, 6:8] = 1.5 * yaw
        return depth_out[:, :-2]

    return depth_out if torch.is_tensor(depth_out) else None


def run_policy_step(env, ppo_runner, policy, depth_encoder, infos, obs):
    depth_latent = compute_depth_latent(env, ppo_runner, depth_encoder, infos, obs)
    with torch.no_grad():
        if hasattr(ppo_runner.alg, "depth_actor"):
            actions = ppo_runner.alg.depth_actor(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
        else:
            actions = policy(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
    return env.step(actions.detach())


def play(args):
    if args.seq_capture_mode == "camera":
        args.headless = True
    elif args.headless:
        raise RuntimeError("motion_sequence.py captures viewer screenshots, so run it without --headless.")

    faulthandler.enable()
    capture_steps = build_capture_steps(args)
    total_steps = args.seq_total_steps if args.seq_total_steps is not None else capture_steps[-1]
    total_steps = max(total_steps, capture_steps[-1])

    log_pth = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", args.proj_name, args.exptid)
    if not os.path.isdir(log_pth):
        raise FileNotFoundError(f"Log directory not found: {log_pth}. Pass --proj_name and --exptid for a trained run.")

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    apply_visualization_cfg(env_cfg, args)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.debug_viz = args.seq_debug_viz
    obs = env.get_observations()
    camera_handle = create_sequence_camera(env, args) if args.seq_capture_mode == "camera" else None

    train_cfg.runner.resume = True
    ppo_runner, train_cfg, loaded_log_dir = task_registry.make_alg_runner(
        log_root=log_pth,
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        return_log_dir=True,
    )
    policy = ppo_runner.get_inference_policy(device=env.device)
    depth_encoder = None
    if env.cfg.depth.use_camera:
        depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)

    output_path = resolve_output_path(args.seq_output, loaded_log_dir)
    frame_dir = os.path.join(os.path.dirname(output_path), os.path.splitext(os.path.basename(output_path))[0] + "_frames")
    os.makedirs(frame_dir, exist_ok=True)

    camera_anchor_pos = None

    infos = {}
    infos["depth"] = env.depth_buffer.clone().to(ppo_runner.device)[:, -1] if getattr(ppo_runner, "if_depth", False) else None

    frame_infos = []
    capture_set = set(capture_steps)
    if 0 in capture_set:
        frame_path = os.path.join(frame_dir, "frame_000000.png")
        camera_anchor_pos = capture_frame(env, args, frame_path, camera_handle, camera_anchor_pos)
        frame_infos.append((frame_path, 0))

    for step in tqdm(range(1, total_steps + 1), desc="capturing sequence"):
        obs, _, _, _, infos = run_policy_step(env, ppo_runner, policy, depth_encoder, infos, obs)
        if step in capture_set:
            frame_path = os.path.join(frame_dir, f"frame_{step:06d}.png")
            camera_anchor_pos = capture_frame(env, args, frame_path, camera_handle, camera_anchor_pos)
            frame_infos.append((frame_path, step))

    compose_sequence_image(frame_infos, output_path, args, env.dt)
    print(f"Saved sequence image: {output_path}")
    print(f"Saved source frames: {frame_dir}")


if __name__ == "__main__":
    sequence_args = parse_sequence_args(None)
    args = get_args()
    for k, v in vars(sequence_args).items():
        setattr(args, k, v)
    play(args)
