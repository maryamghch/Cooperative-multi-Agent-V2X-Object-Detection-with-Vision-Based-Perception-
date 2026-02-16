#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transform every agent's LiDAR points into the ego LiDAR frame across all timestamps,
for all scenarios, then save per-agent transformed clouds and a fused cloud.

Example:
  python scripts/transform_lidar_to_ego.py \
    --data-root /path/to/data/train \
    --pc-root   /path/to/data/train \
    --out-root  /path/to/fused_points/train
"""

import argparse
import yaml
import numpy as np
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

# -------------------------
# Defaults (portable)
# -------------------------
DEFAULT_EGO_IDS = [0, 1, 2, 3]
POSE_KEY_PRIMARY = "lidar_pose"
POSE_KEY_FALLBACK = "true_ego_pose"

# Crop in ego LiDAR coordinates (meters).
DEFAULT_CAV_LIDAR_RANGE = [-30, -15, -5, 30, 15, 5]  # [xmin,ymin,zmin,xmax,ymax,zmax]


# =========================
# Transformer
# =========================
def x_to_world(pose: List[float]) -> np.ndarray:
    """
    Transform from local frame x to CARLA world.
    pose = [x, y, z, roll, yaw, pitch]  (degrees for angles; CARLA axes)
    """
    x, y, z, roll, yaw, pitch = pose[:]

    c_y = np.cos(np.radians(yaw));   s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll));  s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch)); s_p = np.sin(np.radians(pitch))

    M = np.identity(4, dtype=np.float64)
    M[0, 3], M[1, 3], M[2, 3] = x, y, z

    M[0, 0] = c_p * c_y
    M[0, 1] = c_y * s_p * s_r - s_y * c_r
    M[0, 2] = -c_y * s_p * c_r - s_y * s_r
    M[1, 0] = s_y * c_p
    M[1, 1] = s_y * s_p * s_r + c_y * c_r
    M[1, 2] = -s_y * s_p * c_r + c_y * s_r
    M[2, 0] = s_p
    M[2, 1] = -c_p * s_r
    M[2, 2] = c_p * c_r
    return M


def x1_to_x2(x1, x2) -> np.ndarray:
    """
    Transform matrix from frame x1 to frame x2.
    x1, x2 can be poses (list) or already world transforms (np.ndarray).
    """
    if isinstance(x1, list) and isinstance(x2, list):
        x1_to_w = x_to_world(x1)
        x2_to_w = x_to_world(x2)
        w_to_x2 = np.linalg.inv(x2_to_w)
        return w_to_x2 @ x1_to_w
    elif isinstance(x1, list) and not isinstance(x2, list):
        x1_to_w = x_to_world(x1)
        return x2 @ x1_to_w
    else:
        w_to_x2 = np.linalg.inv(x2)
        return w_to_x2 @ x1


# =========================
# IO helpers
# =========================
def read_pose_from_yaml(yaml_path: Path,
                       pose_key_primary: str,
                       pose_key_fallback: str) -> Tuple[List[float], str]:
    """
    Read preferred pose; fallback if missing. Returns ([x,y,z,roll,yaw,pitch], key_used)
    """
    with open(yaml_path, "r") as f:
        y = yaml.safe_load(f) or {}

    if pose_key_primary in y:
        key = pose_key_primary
    elif pose_key_fallback in y:
        key = pose_key_fallback
    else:
        raise KeyError(f"{yaml_path} has neither '{pose_key_primary}' nor '{pose_key_fallback}'")

    p = y[key]
    if isinstance(p, dict):
        pose = [p["x"], p["y"], p.get("z", 0.0), p.get("roll", 0.0), p.get("yaw", 0.0), p.get("pitch", 0.0)]
    else:
        pose = list(p)[:6]

    return [float(v) for v in pose], key


def load_bin_xyzi(path: Path) -> np.ndarray:
    """
    Load KITTI-style .bin (float32). Accepts Nx3 or Nx4. Returns Nx4 (x,y,z,i).
    """
    raw = np.fromfile(str(path), dtype=np.float32)
    if raw.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    if raw.size % 4 == 0:
        pts = raw.reshape(-1, 4)
    elif raw.size % 3 == 0:
        xyz = raw.reshape(-1, 3)
        i = np.zeros((xyz.shape[0], 1), dtype=np.float32)
        pts = np.hstack([xyz, i])
    else:
        raise ValueError(f"Unexpected float count in {path}: {raw.size}")
    return pts.astype(np.float32, copy=False)


def save_bin_xyzi(path: Path, pts: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    pts.astype(np.float32).tofile(str(path))


# =========================
# Geometry
# =========================
def apply_se3(T: np.ndarray, pts_xyzi: np.ndarray) -> np.ndarray:
    """
    Apply 4x4 SE(3) to Nx4 (x,y,z,i) → Nx4 in target frame.
    """
    if pts_xyzi.shape[0] == 0:
        return pts_xyzi
    xyz1 = np.c_[pts_xyzi[:, :3].astype(np.float64),
                 np.ones((pts_xyzi.shape[0], 1), dtype=np.float64)]
    xyz_t = (T @ xyz1.T).T[:, :3]
    out = np.concatenate([xyz_t.astype(np.float32),
                          pts_xyzi[:, 3:4].astype(np.float32)], axis=1)
    return out


def crop_range(pts_xyzi: np.ndarray, rng: List[float]) -> np.ndarray:
    if pts_xyzi.shape[0] == 0:
        return pts_xyzi
    xmin, ymin, zmin, xmax, ymax, zmax = rng
    x, y, z = pts_xyzi[:, 0], pts_xyzi[:, 1], pts_xyzi[:, 2]
    m = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax) & (z >= zmin) & (z <= zmax)
    return pts_xyzi[m]


# =========================
# Utilities
# =========================
def list_timestamps(scen_dir_yaml: Path, ego_id: int) -> List[str]:
    """
    Use the ego YAML folder to drive the timestamp list.
    """
    ego_dir = scen_dir_yaml / str(ego_id)
    if not ego_dir.is_dir():
        return []
    stamps = [p.stem for p in ego_dir.iterdir() if p.suffix == ".yaml"]
    return sorted(stamps)


# =========================
# MAIN
# =========================
def process_scenario(scen_name: str,
                     data_root: Path,
                     pc_root: Path,
                     out_root: Path,
                     ego_ids: List[int],
                     cav_lidar_range: List[float],
                     save_per_agent: bool,
                     save_fused: bool,
                     pose_key_primary: str,
                     pose_key_fallback: str):

    scen_yaml = data_root / scen_name
    scen_pc = pc_root / scen_name
    if not scen_yaml.is_dir() or not scen_pc.is_dir():
        print(f"[warn] Skipping scenario {scen_name}: missing YAML or PC dir")
        return

    agents = sorted([int(d.name) for d in scen_pc.iterdir() if d.is_dir() and d.name.isdigit()])
    if not agents:
        print(f"[warn] No agent folders in {scen_pc}, skipping.")
        return

    print(f"\n[info] === Scenario: {scen_name} | agents={agents} ===")
    print(f"[info] pose_key primary='{pose_key_primary}' (fallback='{pose_key_fallback}')")
    print(f"[info] crop = {cav_lidar_range}")

    for ego_id in ego_ids:
        stamps = list_timestamps(scen_yaml, ego_id)
        if not stamps:
            print(f"[warn] No timestamps under {scen_yaml}/{ego_id}, skipping ego {ego_id}.")
            continue

        out_dir = out_root / scen_name / f"ego_{ego_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        desc = f"{scen_name} → ego {ego_id}"
        for ts in tqdm(stamps, desc=desc):
            ego_yaml = scen_yaml / str(ego_id) / f"{ts}.yaml"
            if not ego_yaml.exists():
                continue

            try:
                ego_pose, _ = read_pose_from_yaml(ego_yaml, pose_key_primary, pose_key_fallback)
            except Exception as e:
                print(f"[err] Failed to read ego pose {ego_yaml}: {e}")
                continue

            fused_buf = []

            for aid in agents:
                ayml = scen_yaml / str(aid) / f"{ts}.yaml"
                abin = scen_pc / str(aid) / f"{ts}.bin"
                if not (ayml.exists() and abin.exists()):
                    continue

                try:
                    ag_pose, _ = read_pose_from_yaml(ayml, pose_key_primary, pose_key_fallback)
                except Exception as e:
                    print(f"[err] Failed to read agent {aid} pose {ayml}: {e}")
                    continue

                T_a2e = x1_to_x2(ag_pose, ego_pose)  # agent -> ego

                pts = load_bin_xyzi(abin)
                pts_e = apply_se3(T_a2e, pts)
                pts_e = crop_range(pts_e, cav_lidar_range)

                if save_per_agent:
                    save_bin_xyzi(out_dir / f"{ts}_from_{aid}.bin", pts_e)

                fused_buf.append(pts_e)

            if save_fused and fused_buf:
                fused = np.concatenate(fused_buf, axis=0)
                save_bin_xyzi(out_dir / f"{ts}_fused.bin", fused)

        print(f"[info] Done scenario={scen_name}, ego={ego_id} → {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True, help="Root folder containing scenario YAML folders")
    p.add_argument("--pc-root", type=Path, required=True, help="Root folder containing scenario pointcloud (.bin) folders")
    p.add_argument("--out-root", type=Path, required=True, help="Output folder to write transformed/fused .bin files")

    p.add_argument("--ego-ids", type=int, nargs="+", default=DEFAULT_EGO_IDS, help="Ego agent IDs to process")
    p.add_argument("--cav-lidar-range", type=float, nargs=6, default=DEFAULT_CAV_LIDAR_RANGE,
                   metavar=("XMIN","YMIN","ZMIN","XMAX","YMAX","ZMAX"),
                   help="Crop range in ego LiDAR frame (meters)")

    p.add_argument("--save-per-agent", action="store_true", help="Save transformed per-agent point clouds")
    p.add_argument("--no-save-per-agent", dest="save_per_agent", action="store_false")
    p.set_defaults(save_per_agent=True)

    p.add_argument("--save-fused", action="store_true", help="Save fused point cloud")
    p.add_argument("--no-save-fused", dest="save_fused", action="store_false")
    p.set_defaults(save_fused=True)

    p.add_argument("--pose-key-primary", default=POSE_KEY_PRIMARY, help="Primary YAML pose key")
    p.add_argument("--pose-key-fallback", default=POSE_KEY_FALLBACK, help="Fallback YAML pose key")

    return p.parse_args()


def main():
    args = parse_args()

    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Missing --data-root: {args.data_root}")
    if not args.pc_root.is_dir():
        raise FileNotFoundError(f"Missing --pc-root: {args.pc_root}")

    scen_dirs = sorted([d for d in args.data_root.iterdir() if d.is_dir()])
    if not scen_dirs:
        print(f"[err] No scenario dirs found under {args.data_root}")
        return

    print(f"[info] Found {len(scen_dirs)} scenarios under {args.data_root}")
    for scen_path in scen_dirs:
        process_scenario(
            scen_name=scen_path.name,
            data_root=args.data_root,
            pc_root=args.pc_root,
            out_root=args.out_root,
            ego_ids=args.ego_ids,
            cav_lidar_range=list(args.cav_lidar_range),
            save_per_agent=args.save_per_agent,
            save_fused=args.save_fused,
            pose_key_primary=args.pose_key_primary,
            pose_key_fallback=args.pose_key_fallback,
        )

    print("All scenarios processed.")


if __name__ == "__main__":
    main()
