#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os, sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch

# -----------------------
# helpers
# -----------------------
def load_bin_xyzit(path: str) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    if raw.size % 4 == 0:
        pts = raw.reshape(-1, 4)
    elif raw.size % 3 == 0:
        xyz = raw.reshape(-1, 3)
        inten = np.zeros((xyz.shape[0], 1), dtype=np.float32)
        pts = np.hstack([xyz, inten])
    else:
        raise ValueError(f"Invalid float count in {path}: {raw.size}")
    m = np.isfinite(pts[:, :3]).all(axis=1)
    return pts[m].astype(np.float32, copy=False)

def mask_points_by_range(points: np.ndarray, limit_range: np.ndarray) -> np.ndarray:
    x_min, y_min, z_min, x_max, y_max, z_max = limit_range
    m = (points[:,0] >= x_min) & (points[:,0] <= x_max) & \
        (points[:,1] >= y_min) & (points[:,1] <= y_max) & \
        (points[:,2] >= z_min) & (points[:,2] <= z_max)
    return points[m]

def mask_ego_points(points: np.ndarray) -> np.ndarray:
    mx = (points[:,0] >= -1.95) & (points[:,0] <=  2.95)
    my = (points[:,1] >= -1.10) & (points[:,1] <=  1.10)
    mz = (points[:,2] >= -2.00) & (points[:,2] <=  1.50)
    return points[~(mx & my & mz)]

def z_clip(points: np.ndarray, zmin: float, zmax: float) -> np.ndarray:
    return points[(points[:,2] >= zmin) & (points[:,2] <= zmax)]

def timestamp_from_fused_name(fused_path: Path) -> str:
    name = fused_path.stem
    return name[:-6] if name.endswith("_fused") else name

def iter_fused_bins(fused_root: Path, scenarios_filter=None, egos_filter=None):
    scen_dirs = sorted([p for p in fused_root.iterdir() if p.is_dir()])
    for sdir in scen_dirs:
        if scenarios_filter and sdir.name not in scenarios_filter:
            continue
        ego_dirs = sorted([p for p in sdir.iterdir() if p.is_dir() and p.name.startswith('ego_')])
        for edir in ego_dirs:
            try:
                ego_id = int(edir.name.split('_', 1)[1])
            except Exception:
                continue
            if egos_filter and ego_id not in egos_filter:
                continue
            fused_bins = sorted(list(edir.glob("*_fused.bin")))
            if not fused_bins:
                fused_bins = sorted(list(edir.rglob("*_fused.bin")))
            for fb in fused_bins:
                yield sdir.name, ego_id, fb

def parse_args():
    ap = argparse.ArgumentParser()

    # paths
    ap.add_argument("--pcdet-root", type=Path, required=True,
                    help="Path to OpenPCDet repo root (contains pcdet/)")
    ap.add_argument("--cfg-file", type=Path, required=True, help="Path to OpenPCDet model YAML config")
    ap.add_argument("--ckpt-file", type=Path, required=True, help="Path to OpenPCDet checkpoint .pth")

    ap.add_argument("--fused-root", type=Path, required=True,
                    help="Root folder with <scenario>/ego_<id>/*_fused.bin")
    ap.add_argument("--save-root", type=Path, required=True,
                    help="Where to save outputs (npy)")

    # filters
    ap.add_argument("--scenarios", nargs="*", default=None, help="Optional scenario name filter list")
    ap.add_argument("--egos", type=int, nargs="*", default=None, help="Optional ego id filter list")

    # options
    ap.add_argument("--remove-ego-box", action="store_true", help="Remove ego-vehicle points")
    ap.add_argument("--no-remove-ego-box", dest="remove_ego_box", action="store_false")
    ap.set_defaults(remove_ego_box=True)

    ap.add_argument("--extra-z-clip", type=float, nargs=2, default=None,
                    metavar=("ZMIN","ZMAX"), help="Optional extra z filtering")

    ap.add_argument("--save-dtype", default="float16", choices=["float16","float32"],
                    help="dtype for saving feature arrays")

    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"],
                    help="Device selection")

    return ap.parse_args()

def main():
    args = parse_args()

    # --- sanity checks
    if not (args.pcdet_root / "pcdet").is_dir():
        raise FileNotFoundError(f"--pcdet-root must contain pcdet/: {args.pcdet_root}")
    if not args.cfg_file.exists():
        raise FileNotFoundError(f"Missing --cfg-file: {args.cfg_file}")
    if not args.ckpt_file.exists():
        raise FileNotFoundError(f"Missing --ckpt-file: {args.ckpt_file}")
    if not args.fused_root.is_dir():
        raise FileNotFoundError(f"Missing --fused-root: {args.fused_root}")

    args.save_root.mkdir(parents=True, exist_ok=True)

    # --- add OpenPCDet to PYTHONPATH (portable)
    sys.path.insert(0, str(args.pcdet_root))

    from pcdet.models import build_network
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.utils import common_utils
    from pcdet.datasets.kitti.kitti_dataset import KittiDataset
    from spconv.pytorch.utils import PointToVoxel

    # device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Init PCDet
    cfg_from_yaml_file(str(args.cfg_file), cfg)
    logger = common_utils.create_logger()

    dataset_template = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=Path(args.fused_root),
        logger=logger
    )

    point_cloud_range = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE, dtype=np.float32)
    voxel_size = None
    max_points_per_voxel = 64
    max_voxels = 200000

    for proc in cfg.DATA_CONFIG.DATA_PROCESSOR:
        if proc.NAME == 'transform_points_to_voxels':
            voxel_size = np.array(proc.VOXEL_SIZE, dtype=np.float32)
            max_points_per_voxel = int(proc.get('MAX_POINTS_PER_VOXEL', 32))
            mv = proc.get('MAX_NUMBER_OF_VOXELS', {'train': 200000, 'test': 200000})
            max_voxels = int(mv['test'] if isinstance(mv, dict) else mv)
            break
    assert voxel_size is not None, "VOXEL_SIZE not found in cfg.DATA_CONFIG.DATA_PROCESSOR"

    H_bev = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])
    W_bev = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
    print(f"[cfg] BEV grid (pre-backbone): H={H_bev}, W={W_bev}; voxel={voxel_size.tolist()} range={point_cloud_range.tolist()} max_voxels={max_voxels}")

    voxel_generator = PointToVoxel(
        vsize_xyz=voxel_size.tolist(),
        coors_range_xyz=point_cloud_range.tolist(),
        num_point_features=4,
        max_num_points_per_voxel=max_points_per_voxel,
        max_num_voxels=max_voxels,
        device=device
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset_template)
    model.load_params_from_file(filename=str(args.ckpt_file), logger=logger, to_cpu=(device.type == 'cpu'))
    model.to(device).eval()

    save_dtype = np.float16 if args.save_dtype == "float16" else np.float32

    any_found = False
    for scen_name, ego_id, fused_bin in iter_fused_bins(args.fused_root, args.scenarios, args.egos):
        any_found = True
        ts = timestamp_from_fused_name(fused_bin)

        out_dir = args.save_root / scen_name / str(ego_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{ts}_occ.npy"

        try:
            points = load_bin_xyzit(str(fused_bin))
            if points.shape[0] == 0:
                print(f" Empty after load: {fused_bin}")
                continue

            if args.extra_z_clip is not None:
                points = z_clip(points, args.extra_z_clip[0], args.extra_z_clip[1])
                if points.size == 0:
                    print(f" All filtered by Z in {fused_bin}")
                    continue

            points = mask_points_by_range(points, point_cloud_range)
            if points.size == 0:
                print(f" All out of POINT_CLOUD_RANGE in {fused_bin}")
                continue

            if args.remove_ego_box:
                points = mask_ego_points(points)
                if points.size == 0:
                    print(f" All removed by ego-box mask in {fused_bin}")
                    continue

            xmn, xmx = float(points[:,0].min()), float(points[:,0].max())
            ymn, ymx = float(points[:,1].min()), float(points[:,1].max())
            print(f"[{scen_name} ego{ego_id} {ts}] pts={points.shape[0]} x[{xmn:.1f},{xmx:.1f}] y[{ymn:.1f},{ymx:.1f}]")

            pts_t = torch.from_numpy(points).to(device=device, dtype=torch.float32)
            voxels, coords, num_points = voxel_generator(pts_t)
            if voxels.shape[0] == 0:
                print(f" No voxels created for {fused_bin}")
                continue

            coords = coords.to(device=device, dtype=torch.int32)
            batch_idx = torch.zeros((coords.shape[0], 1), dtype=torch.int32, device=coords.device)
            coordinates = torch.cat([batch_idx, coords], dim=1)  # [M,4]=[b,z,y,x]

            # Occupancy grid from voxel indices (y,x)
            occ = torch.zeros((H_bev, W_bev), dtype=torch.int32, device=device)
            yy = coordinates[:, 2].to(torch.long)
            xx = coordinates[:, 3].to(torch.long)
            yy.clamp_(0, H_bev - 1)
            xx.clamp_(0, W_bev - 1)
            val = torch.ones(yy.shape[0], dtype=torch.int32, device=device)
            occ.index_put_((yy, xx), val, accumulate=True)

            np.save(str(out_path), occ.cpu().numpy().astype(save_dtype))

        except Exception as e:
            print(f" Error {fused_bin}: {e}")
            continue

    if not any_found:
        print(f" No fused bins found under {args.fused_root}. Expecting <scenario>/ego_<id>/*_fused.bin")
    else:
        print(" Done: occupancy extracted from fused bins.")

if __name__ == "__main__":
    main()
