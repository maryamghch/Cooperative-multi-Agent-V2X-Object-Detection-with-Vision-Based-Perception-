#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import csv
import math
import random
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import yaml
import numpy as np
import cv2
import optuna
from ultralytics import YOLO


# =========================
# PATHS / DEFAULTS (portable)
# =========================
REPO_ROOT = Path(__file__).resolve().parents[1]  # repo/scripts/this_file.py -> repo/

DEFAULT_DATA_ROOT = REPO_ROOT / "data"  # expects data/<split>/<scene>/<agent>/*.yaml etc.
DEFAULT_BEV_ROOT  = REPO_ROOT / "BEV_Features_Final_FUSED"  # expects BEV_Features_Final_FUSED/<split>/<scene>/<ego>/<stamp>_occ.npy
DEFAULT_CFG_FILE  = REPO_ROOT / "configs" / "pointpillar.yaml"

DEFAULT_YOLO_CACHE_DIR = REPO_ROOT / "cache_yolo_optuna_raw"
DEFAULT_OUT_DIR = REPO_ROOT / "tuning_out"


# =========================
# PARAMS
# =========================
@dataclass
class Params:
    # YOLO / classes
    MODEL_YOLO: str = "yolov8x.pt"
    YOLO_VEHICLE_CLASSES: Tuple[str, ...] = ("car", "van", "truck", "bus")

    # tuned
    DET_CONF: float = 0.30
    DET_IOU: float = 0.40
    USE_CLASS_AGNOSTIC_NMS: bool = True
    NMS_IOU_AGNOSTIC: float = 0.70

    # LiDAR projection / filtering
    MIN_Z_CAM_VIS: float = 0.20
    USE_BOTTOM_STRIP: bool = True
    BOTTOM_STRIP_FRAC: float = 0.40
    MIN_LIDAR_PTS_IN_BOX: int = 6
    MIN_RANGE_KEEP: float = 1.5
    MAX_RANGE_KEEP: float = 120.0

    # Center clustering
    USE_CLUSTER_FOR_CENTER: bool = True
    CLUSTER_EPS_M: float = 0.6
    CLUSTER_MIN_PTS: int = 8

    # =========================
    # Yaw (PCA + strong ground filtering + XY trimming)
    # =========================
    YAW_USE_FULL_BOX: bool = True

    # Force PCA-only yaw
    USE_MINAREARECT_YAW: bool = False
    USE_PCA_FALLBACK_YAW: bool = True

    # points / PCA confidence
    MIN_YAW_PTS: int = 22
    SHAPE_EIG_RATIO_MIN: float = 2.2

    # strong ground filtering
    GROUND_Z_PERCENTILE: int = 20
    GROUND_Z_MARGIN: float = 0.30

    # XY trimming to suppress curb/wall/ground-line dominance before PCA
    TRIM_YAW_XY: bool = True
    TRIM_YAW_KEEP_SCALE: float = 0.80
    TRIM_YAW_SLACK_M: float = 0.50
    TRIM_YAW_MIN_KEEP: int = 12

    # Optional 90-deg disambiguation
    USE_CLASS_PRIOR_90_DISAMBIG: bool = True
    YAW_OFFSET_DEG: float = 0.0

    # =========================
    # Fusion / pairing
    # =========================
    # merge by rotated BEV overlap (intersection / min(area))
    MERGE_OVERLAP_MIN: float = 0.10   # 10% overlap -> merge
    PAIR_GATE_M: float = 6.0

    # post-suppress (keep one per small neighborhood after merging)
    FUSE_POST_SUPPRESS: bool = True
    FUSE_SUPPRESS_DIST_M: float = 1.2
    FUSE_SUPPRESS_KEEP_BY: str = "score"  # "score" or "support" or "conf"
    FUSE_WEIGHT_BY_SUPPORT: bool = True

    # ===== Yaw borrowing =====
    BORROW_YAW_MAX_DIST_M: float = 3.0  # meters, tuned by Optuna

    # ===== Self-detection removal (YOLO-stage mask) =====
    # Apply ONLY for agents 0 and 1
    REMOVE_SELF_FROM_YOLO: bool = True
    SELF_MASK_IOU_THR: float = 0.20
    SELF_MASK_AGENTS: Tuple[str, ...] = ("0", "1")

    # Per-camera ego-hood mask as fractions of (W,H): (x1f,y1f,x2f,y2f)
    EGO_HOOD_MASK_FRAC: Optional[Dict[str, Tuple[float, float, float, float]]] = None

    # BEV
    SWAP_XY_OCC: bool = False

    # GT
    GT_CLUSTER_THR_M: float = 1.0
    VALID_GT_CLASSES: Tuple[str, ...] = ("car", "van", "truck", "bus", "concretetruck")

    def __post_init__(self):
        if self.EGO_HOOD_MASK_FRAC is None:
            self.EGO_HOOD_MASK_FRAC = {
                "cam1": (0.00, 0.78, 1.00, 1.00),
                "cam2": (0.00, 0.78, 1.00, 1.00),
                "cam3": (0.00, 0.78, 1.00, 1.00),
                "cam4": (0.00, 0.78, 1.00, 1.00),
            }


SIZE_BY_CLASS = {
    "car":   (4.5, 2.0),
    "van":   (5.2, 2.1),
    "truck": (8.0, 2.5),
    "bus":   (10.5, 2.6),
    "concretetruck": (9.5, 2.6),
}


# =========================
# ANGLES
# =========================
def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def ang_diff_axis(a: float, b: float) -> float:
    d = abs(wrap_pi(a - b))
    return min(d, math.pi - d)


# =========================
# CFG ranges (OpenPCDet)
# =========================
def load_cfg_ranges(cfg_path: Path):
    cfg = yaml.safe_load(open(cfg_path, "r"))
    pcr = np.array(cfg["DATA_CONFIG"]["POINT_CLOUD_RANGE"], dtype=np.float64)
    voxel = None
    for p in cfg["DATA_CONFIG"]["DATA_PROCESSOR"]:
        if p.get("NAME", "") == "transform_points_to_voxels":
            voxel = np.array(p["VOXEL_SIZE"], dtype=np.float64)
            break
    if voxel is None:
        raise RuntimeError("VOXEL_SIZE not found in cfg")
    return pcr, voxel

def ego_to_pix_float(x, y, pcr, voxel, swap_xy=False):
    if swap_xy:
        x, y = y, x
    xx = (x - pcr[0]) / voxel[0]
    yy = (y - pcr[1]) / voxel[1]
    return float(xx), float(yy)

def is_inside_bev_xy(x_m, y_m, pcr, voxel, occ_shape=None, swap_xy=False, margin_pix=0.0):
    if occ_shape is not None:
        H, W = int(occ_shape[0]), int(occ_shape[1])
        xx, yy = ego_to_pix_float(x_m, y_m, pcr, voxel, swap_xy=swap_xy)
        return (xx >= margin_pix) and (yy >= margin_pix) and (xx < (W - margin_pix)) and (yy < (H - margin_pix))
    x_min, y_min, _, x_max, y_max, _ = [float(v) for v in pcr.tolist()]
    return (x_m >= x_min) and (x_m <= x_max) and (y_m >= y_min) and (y_m <= y_max)


# =========================
# YAML / LiDAR IO
# =========================
def load_yaml(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_sensor_pose_for_dets(meta):
    if "lidar_pose" in meta and meta["lidar_pose"] is not None:
        return meta["lidar_pose"]
    return meta["true_ego_pose"]

def find_lidar_file(agent_dir: Path, stamp: str):
    candidates = [
        agent_dir / f"{stamp}.bin",
        agent_dir / f"{stamp}_lidar.bin",
        agent_dir / f"lidar_{stamp}.bin",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

def load_lidar_bin(bin_path: Path, stride=4, dtype=np.float32):
    arr = np.fromfile(str(bin_path), dtype=dtype)
    if arr.size % stride != 0:
        raise ValueError(f"Unexpected LiDAR bin size: {bin_path} (size={arr.size})")
    pts = arr.reshape(-1, stride)[:, :3]
    return pts.astype(np.float64)

def load_metas_for_stamp(scene_dir: Path, agents, stamp: str):
    metas = {}
    for a in agents:
        yp = scene_dir / a / f"{stamp}.yaml"
        if yp.exists():
            metas[a] = load_yaml(yp)
    return metas

def load_occ_shape(bev_root: Path, split: str, scene: str, ego_agent: str, stamp: str):
    occ_path = bev_root / split / scene / ego_agent / f"{stamp}_occ.npy"
    if occ_path.exists():
        try:
            occ = np.load(occ_path, mmap_mode="r")
            return tuple(occ.shape)
        except Exception:
            return None
    return None


# =========================
# SE(2)
# =========================
def yaw_deg_pose(p):  # [x,y,z,roll,yaw,pitch]
    return float(p[4])

def T_world_from_pose_se2(p):
    x, y = float(p[0]), float(p[1])
    yaw = math.radians(yaw_deg_pose(p))
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, x],
                     [s,  c, y],
                     [0,  0, 1]], dtype=np.float64)

def se2_apply(T, x, y):
    q = T @ np.array([x, y, 1.0], dtype=np.float64)
    return float(q[0]), float(q[1])


# =========================
# Classes
# =========================
def normalize_gt_class(obj_type):
    if obj_type is None:
        return ""
    s = str(obj_type).strip().lower()
    s = s.replace("vehicle.", "").replace("carla.", "")
    s = s.replace(" ", "").replace("-", "").replace("_", "")
    if s in {"concretetruck", "cementtruck"}:
        return "concretetruck"
    return s


# =========================
# Projection (extrinsic assumed CAM->LiDAR)
# =========================
def to_4x4(m):
    m = np.array(m, dtype=np.float64)
    if m.shape == (4, 4):
        return m
    if m.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = m
        return out
    raise ValueError(f"Unexpected extrinsic shape: {m.shape}")

def project_lidar_to_image(points_lidar_xyz, K, T_cam2lidar, min_z_cam=0.3):
    T_lidar2cam = np.linalg.inv(T_cam2lidar)

    N = points_lidar_xyz.shape[0]
    P = np.ones((N, 4), dtype=np.float64)
    P[:, :3] = points_lidar_xyz

    Pc = (T_lidar2cam @ P.T).T
    Xc, Yc, Zc = Pc[:, 0], Pc[:, 1], Pc[:, 2]

    valid = np.isfinite(Zc) & (Zc > float(min_z_cam))

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    u = fx * (Xc / (Zc + 1e-12)) + cx
    v = fy * (Yc / (Zc + 1e-12)) + cy
    valid &= np.isfinite(u) & np.isfinite(v)

    return u, v, Zc, valid

def lidar_points_in_box(u, v, Zc, valid, x1, y1, x2, y2, H, W, params: Params, use_bottom_strip=True):
    x1c = max(0.0, min(float(W - 1), float(x1)))
    x2c = max(0.0, min(float(W - 1), float(x2)))
    y1c = max(0.0, min(float(H - 1), float(y1)))
    y2c = max(0.0, min(float(H - 1), float(y2)))
    if x2c <= x1c or y2c <= y1c:
        return np.zeros((u.shape[0],), dtype=bool)

    if use_bottom_strip:
        y_strip0 = y2c - params.BOTTOM_STRIP_FRAC * (y2c - y1c)
        y1_use, y2_use = y_strip0, y2c
    else:
        y1_use, y2_use = y1c, y2c

    m = valid.copy()
    m &= (u >= x1c) & (u <= x2c) & (v >= y1_use) & (v <= y2_use)
    m &= np.isfinite(Zc)
    m &= (Zc >= params.MIN_RANGE_KEEP) & (Zc <= params.MAX_RANGE_KEEP)
    return m


# =========================
# Clustering
# =========================
def cluster_xy_indices(pts_xy, eps=0.9, min_pts=8):
    if pts_xy.shape[0] == 0:
        return []
    N = pts_xy.shape[0]
    used = np.zeros((N,), dtype=bool)
    clusters = []
    for i in range(N):
        if used[i]:
            continue
        used[i] = True
        q = [i]
        cl = [i]
        while q:
            k = q.pop()
            dx = pts_xy[:, 0] - pts_xy[k, 0]
            dy = pts_xy[:, 1] - pts_xy[k, 1]
            nn = np.flatnonzero((~used) & (dx*dx + dy*dy <= eps*eps))
            for j in nn.tolist():
                used[j] = True
                q.append(j)
                cl.append(j)
        if len(cl) >= int(min_pts):
            clusters.append(cl)
    return clusters

def pick_best_cluster_indices(clusters):
    if not clusters:
        return None
    return max(clusters, key=lambda c: len(c))


# =========================
# Unique-GT clustering (count each object once)
# =========================
def unique_gt_by_xy(gt_list: List[dict], thr_m: float) -> List[dict]:
    if not gt_list:
        return []

    thr2 = float(thr_m) * float(thr_m)
    N = len(gt_list)
    used = [False] * N
    out = []

    for i in range(N):
        if used[i]:
            continue
        used[i] = True
        cl = [i]
        q = [i]
        while q:
            k = q.pop()
            xk, yk = float(gt_list[k]["x0"]), float(gt_list[k]["y0"])
            for j in range(N):
                if used[j]:
                    continue
                dx = float(gt_list[j]["x0"]) - xk
                dy = float(gt_list[j]["y0"]) - yk
                if dx*dx + dy*dy <= thr2:
                    used[j] = True
                    q.append(j)
                    cl.append(j)

        xs = np.array([float(gt_list[t]["x0"]) for t in cl], dtype=np.float64)
        ys = np.array([float(gt_list[t]["y0"]) for t in cl], dtype=np.float64)
        x0 = float(np.median(xs))
        y0 = float(np.median(ys))

        best = min(cl, key=lambda t: (float(gt_list[t]["x0"]) - x0) ** 2 + (float(gt_list[t]["y0"]) - y0) ** 2)
        rep = dict(gt_list[best])
        rep["x0"] = x0
        rep["y0"] = y0
        out.append(rep)

    return out


# =========================
# Yaw helpers: XY trimming + strong ground filtering
# =========================
def trim_yaw_points_xy_mask(pts_xy: np.ndarray, cls: str, params: Params) -> Optional[np.ndarray]:
    if pts_xy is None or pts_xy.shape[0] < max(5, params.TRIM_YAW_MIN_KEEP):
        return None

    cx = float(np.median(pts_xy[:, 0]))
    cy = float(np.median(pts_xy[:, 1]))
    d2 = (pts_xy[:, 0] - cx) ** 2 + (pts_xy[:, 1] - cy) ** 2

    if cls in SIZE_BY_CLASS:
        L, W = SIZE_BY_CLASS[cls]
        diag = math.sqrt(L * L + W * W)
        r = float(params.TRIM_YAW_KEEP_SCALE) * 0.5 * diag + float(params.TRIM_YAW_SLACK_M)
    else:
        r = 3.0

    keep = d2 <= (r * r)
    if int(keep.sum()) >= int(params.TRIM_YAW_MIN_KEEP):
        return keep
    return None

def filter_ground_by_agent_z(pts_a_xyz, pts_ego_xy, params: Params):
    if pts_a_xyz is None or pts_ego_xy is None:
        return pts_ego_xy
    if pts_a_xyz.shape[0] != pts_ego_xy.shape[0]:
        return pts_ego_xy
    if pts_a_xyz.shape[0] < params.MIN_YAW_PTS:
        return pts_ego_xy

    z = pts_a_xyz[:, 2]
    z0 = np.percentile(z, params.GROUND_Z_PERCENTILE)
    keep = z >= (z0 + params.GROUND_Z_MARGIN)
    if int(keep.sum()) >= params.MIN_YAW_PTS:
        return pts_ego_xy[keep]
    return pts_ego_xy


# =========================
# Yaw estimation (PCA only)
# =========================
def yaw_from_pca(pts_xy, params: Params, last_yaw=None):
    if pts_xy is None or pts_xy.shape[0] < params.MIN_YAW_PTS:
        return None

    X = pts_xy - np.mean(pts_xy, axis=0, keepdims=True)
    C = (X.T @ X) / max(1, X.shape[0] - 1)
    w, V = np.linalg.eigh(C)
    w0, w1 = float(w[0]), float(w[1])

    if (not np.isfinite(w0)) or (not np.isfinite(w1)) or w1 < 1e-9:
        return None

    ratio = w1 / max(w0, 1e-12)
    if ratio < float(params.SHAPE_EIG_RATIO_MIN):
        return None

    v = V[:, 1]
    yaw = wrap_pi(math.atan2(float(v[1]), float(v[0])))

    if last_yaw is not None:
        y0 = yaw
        y1 = wrap_pi(yaw + math.pi)
        yaw = y0 if abs(wrap_pi(y0 - last_yaw)) <= abs(wrap_pi(y1 - last_yaw)) else y1

    return wrap_pi(yaw)

def extent_along_axes(pts_xy, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[ c, s],
                  [-s, c]], dtype=np.float64)
    q = (R @ pts_xy.T).T
    dx = float(q[:, 0].max() - q[:, 0].min())
    dy = float(q[:, 1].max() - q[:, 1].min())
    return dx, dy

def yaw_choose_by_class_prior(pts_xy, yaw0, cls):
    if cls not in SIZE_BY_CLASS or pts_xy is None or pts_xy.shape[0] < 5:
        return yaw0
    Lexp, Wexp = SIZE_BY_CLASS[cls]

    cands = []
    for add in (0.0, math.pi/2):
        y = wrap_pi(yaw0 + add)
        dx, dy = extent_along_axes(pts_xy, y)
        errL = abs(dx - Lexp) / max(Lexp, 1e-6)
        errW = abs(dy - Wexp) / max(Wexp, 1e-6)
        score = errL + errW
        cands.append((score, y))
    cands.sort(key=lambda t: t[0])
    return cands[0][1]


# =========================
# GT to ego
# =========================
def bbox_center_world_from_yaml(obj):
    loc = np.array(obj["location"], dtype=np.float64)
    cen = np.array(obj.get("center", [0, 0, 0]), dtype=np.float64)
    yaw_deg_obj = float(obj["angle"][1])
    yaw = math.radians(yaw_deg_obj)
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return loc - (R @ cen)

def collect_gt_ego(metas_cur, T_ego_from_w, ego_yaw_world, params: Params):
    gt_list = []
    valid = set(params.VALID_GT_CLASSES)
    for _, m in metas_cur.items():
        vehicles = m.get("vehicles", {})
        if not isinstance(vehicles, dict):
            continue
        for _, obj in vehicles.items():
            cls = normalize_gt_class(obj.get("obj_type", ""))
            if cls not in valid:
                continue
            if "location" not in obj or "angle" not in obj:
                continue

            center_w = bbox_center_world_from_yaml(obj)
            xw, yw = float(center_w[0]), float(center_w[1])
            x0, y0 = se2_apply(T_ego_from_w, xw, yw)

            yaw_w = math.radians(float(obj["angle"][1]))
            yaw_ego = wrap_pi(yaw_w - ego_yaw_world)
            yaw_ego = wrap_pi(yaw_ego + math.radians(params.YAW_OFFSET_DEG))

            gt_list.append({"x0": x0, "y0": y0, "yaw_ego": yaw_ego, "cls": cls})
    return gt_list


# =========================
# Pairing + errors
# =========================
def pair_dets_to_gt(dets, gt_list, gate_m=6.0):
    pairs = []
    if not dets or not gt_list:
        return pairs

    used_gt = set()
    for d in dets:
        best_j = None
        best_d = 1e9
        for j, g in enumerate(gt_list):
            if j in used_gt:
                continue
            dist = math.hypot(d["x0"] - g["x0"], d["y0"] - g["y0"])
            if dist < best_d:
                best_d = dist
                best_j = j
        if best_j is not None and best_d <= gate_m:
            used_gt.add(best_j)
            g = gt_list[best_j]
            yaw_err = None
            if (d.get("yaw_ego") is not None) and (g.get("yaw_ego") is not None):
                yaw_err = ang_diff_axis(d["yaw_ego"], g["yaw_ego"])
            pairs.append((d, g, best_d, yaw_err))
    return pairs


# =========================
# NMS (class-aware + class-agnostic)
# =========================
def box_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-12
    return inter / union

def nms_class_agnostic(dets, iou_thr):
    dets = sorted(dets, key=lambda d: -d["conf"])
    keep = []
    for d in dets:
        ok = True
        for k in keep:
            if box_iou_xyxy(d["xyxy"], k["xyxy"]) >= iou_thr:
                ok = False
                break
        if ok:
            keep.append(d)
    return keep

def nms_class_aware(dets, iou_thr):
    by_cls: Dict[str, List[dict]] = {}
    for d in dets:
        by_cls.setdefault(d["cls"], []).append(d)
    out = []
    for _, group in by_cls.items():
        out.extend(nms_class_agnostic(group, iou_thr))
    return out


# =========================
# YOLO SELF-MASK (hood/ego removal) -- only for agents 0 and 1
# =========================
def ego_mask_xyxy(params: Params, cam: str, H: int, W: int):
    if params.EGO_HOOD_MASK_FRAC is None or cam not in params.EGO_HOOD_MASK_FRAC:
        return None
    x1f, y1f, x2f, y2f = params.EGO_HOOD_MASK_FRAC[cam]
    x1 = float(x1f) * (W - 1)
    x2 = float(x2f) * (W - 1)
    y1 = float(y1f) * (H - 1)
    y2 = float(y2f) * (H - 1)
    return (x1, y1, x2, y2)

def is_self_box_yolo(params: Params, agent: str, cam: str, box_xyxy, H: int, W: int):
    if str(agent) not in set(params.SELF_MASK_AGENTS):
        return False
    m = ego_mask_xyxy(params, cam, H, W)
    if m is None:
        return False
    return box_iou_xyxy(box_xyxy, m) >= float(params.SELF_MASK_IOU_THR)


# =========================
# Rotated BEV overlap (merge criterion)
# =========================
def det_to_rotated_rect(d: dict) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    cls = str(d.get("cls", "car"))
    Lm, Wm = SIZE_BY_CLASS.get(cls, (4.8, 2.2))
    yaw = d.get("yaw_ego", 0.0)
    if yaw is None or (not np.isfinite(yaw)):
        yaw = 0.0
    angle_deg = float(math.degrees(float(yaw)))
    return ((float(d["x0"]), float(d["y0"])), (float(Lm), float(Wm)), angle_deg)

def rotated_overlap_min_area(di: dict, dj: dict) -> float:
    ri = det_to_rotated_rect(di)
    rj = det_to_rotated_rect(dj)

    ai = float(ri[1][0] * ri[1][1])
    aj = float(rj[1][0] * rj[1][1])
    amin = max(1e-12, min(ai, aj))

    inter_type, inter_pts = cv2.rotatedRectangleIntersection(ri, rj)
    if inter_type == cv2.INTERSECT_NONE or inter_pts is None:
        return 0.0

    poly = inter_pts.reshape(-1, 2).astype(np.float32)
    if poly.shape[0] < 3:
        return 0.0
    inter_area = float(abs(cv2.contourArea(poly)))
    return float(inter_area / amin)


# =========================
# Fusion (overlap-based, rotated)
# =========================
def fuse_dets_better(dets: List[dict], params: Params) -> List[dict]:
    if not dets:
        return []

    N = len(dets)

    def can_merge(i, j):
        ov = rotated_overlap_min_area(dets[i], dets[j])
        return ov >= float(params.MERGE_OVERLAP_MIN)

    parent = list(range(N))
    rank = [0] * N

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for i in range(N):
        for j in range(i + 1, N):
            if can_merge(i, j):
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(N):
        r = find(i)
        groups.setdefault(r, []).append(i)

    fused = []
    for idxs in groups.values():
        xs, ys, ws = [], [], []
        yaw_sin, yaw_cos, yaw_w = [], [], []
        supports = []

        for k in idxs:
            d = dets[k]
            conf = float(d.get("conf", 0.0))
            sup = float(d.get("support", 1.0))
            wk = conf * (sup if params.FUSE_WEIGHT_BY_SUPPORT else 1.0)
            wk = max(wk, 1e-6)

            xs.append(float(d["x0"]))
            ys.append(float(d["y0"]))
            ws.append(wk)
            supports.append(int(d.get("support", 0)))

            y = d.get("yaw_ego", None)
            if y is not None and np.isfinite(y):
                yaw_sin.append(math.sin(float(y)) * wk)
                yaw_cos.append(math.cos(float(y)) * wk)
                yaw_w.append(wk)

        wsum = float(np.sum(ws)) if ws else 1.0
        x0 = float(np.sum(np.array(xs) * np.array(ws)) / wsum)
        y0 = float(np.sum(np.array(ys) * np.array(ws)) / wsum)

        yaw_ego = None
        if len(yaw_w) >= 1:
            s = float(np.sum(yaw_sin))
            c = float(np.sum(yaw_cos))
            if abs(s) + abs(c) > 1e-9:
                yaw_ego = float(wrap_pi(math.atan2(s, c)))

        best = max(idxs, key=lambda k: float(dets[k].get("conf", 0.0)))
        cls = dets[best].get("cls", "car")
        conf_best = float(dets[best].get("conf", 0.0))
        support_sum = int(np.sum(supports))

        fused.append({
            "cls": cls,
            "x0": x0, "y0": y0,
            "yaw_ego": yaw_ego,
            "conf": conf_best,
            "support": support_sum,
            "n_members": int(len(idxs)),
            "score": float(conf_best * max(1.0, support_sum)),
        })

    # post-suppress
    if (not params.FUSE_POST_SUPPRESS) or len(fused) <= 1:
        return fused

    key = str(params.FUSE_SUPPRESS_KEEP_BY)
    keep = []
    fused_sorted = sorted(fused, key=lambda d: -float(d.get(key, d.get("score", 0.0))))
    for d in fused_sorted:
        ok = True
        for k in keep:
            if math.hypot(d["x0"] - k["x0"], d["y0"] - k["y0"]) <= float(params.FUSE_SUPPRESS_DIST_M):
                ok = False
                break
        if ok:
            keep.append(d)
    return keep


# =========================
# YAW BORROWING
# =========================
def borrow_yaw_from_neighbors(fused: List[dict], max_dist: float) -> List[dict]:
    if not fused or not np.isfinite(max_dist) or max_dist <= 0.0:
        return fused

    have_yaw_idx = [
        i for i, d in enumerate(fused)
        if (d.get("yaw_ego") is not None) and np.isfinite(d.get("yaw_ego", np.nan))
    ]
    if not have_yaw_idx:
        return fused

    max_d = float(max_dist)
    for i, d in enumerate(fused):
        y = d.get("yaw_ego")
        if (y is not None) and np.isfinite(y):
            continue

        best_j = None
        best_dist = max_d
        for j in have_yaw_idx:
            dj = fused[j]
            dist = math.hypot(d["x0"] - dj["x0"], d["y0"] - dj["y0"])
            if dist <= best_dist:
                best_dist = dist
                best_j = j

        if best_j is not None:
            src_yaw = fused[best_j].get("yaw_ego")
            if src_yaw is not None and np.isfinite(src_yaw):
                d["yaw_ego"] = float(src_yaw)
                d["yaw_borrowed"] = True

    return fused


# =========================
# YOLO RAW CACHE
# =========================
def yolo_raw_cache_path(cache_dir: Path, split: str, scene: str, agent: str, stamp: str, cam: str):
    return cache_dir / "raw" / split / scene / agent / f"{stamp}_{cam}.json"

def load_json_silent(p: Path):
    try:
        if p.exists():
            return json.loads(p.read_text())
    except Exception:
        return None
    return None

def save_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))

def run_yolo_raw(yolo_model, img_bgr) -> List[dict]:
    r = yolo_model.predict(img_bgr, conf=0.01, iou=0.90, verbose=False)[0]
    out = []
    if r.boxes is None:
        return out
    names = yolo_model.names
    for b in r.boxes:
        cls = str(names[int(b.cls)]).lower()
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        out.append({
            "cls": cls,
            "xyxy": (float(x1), float(y1), float(x2), float(y2)),
            "conf": float(b.conf),
        })
    return out

def filter_raw_dets_for_params(raw_dets: List[dict], params: Params, agent: str, cam: str, H: int, W: int) -> List[dict]:
    allowed = set([c.lower() for c in params.YOLO_VEHICLE_CLASSES])
    dets = [d for d in raw_dets if (d["cls"] in allowed and float(d["conf"]) >= float(params.DET_CONF))]

    if params.REMOVE_SELF_FROM_YOLO and dets:
        dets = [d for d in dets if not is_self_box_yolo(params, agent, cam, d["xyxy"], H, W)]

    if len(dets) > 1:
        dets = nms_class_aware(dets, float(params.DET_IOU))

    if params.USE_CLASS_AGNOSTIC_NMS and len(dets) > 1:
        dets = nms_class_agnostic(dets, float(params.NMS_IOU_AGNOSTIC))

    return dets


# =========================
# Listing scenes/stamps
# =========================
def list_scenes(data_root: Path):
    if not data_root.exists():
        return []
    scenes = [p.name for p in data_root.iterdir() if p.is_dir()]
    scenes.sort()
    return scenes

def list_stamps(scene_dir: Path, agent: str = "0"):
    d = scene_dir / agent
    if not d.exists():
        return []
    stamps = []
    for yp in d.glob("*.yaml"):
        name = yp.stem
        if name.isdigit():
            stamps.append(name)
    stamps.sort()
    return stamps


# =========================
# Evaluate split
# =========================
def evaluate_split(
    split: str,
    params: Params,
    data_root: Path = DEFAULT_DATA_ROOT,
    bev_root: Path = DEFAULT_BEV_ROOT,
    cfg_file: Path = DEFAULT_CFG_FILE,
    agents=("0","1","2","3"),
    cams=("cam1","cam2","cam3","cam4"),
    ego_agents=("0","1","2","3"),
    scene_limit=None,
    stamp_stride=1,
    seed=0,
    yolo_cache_dir: Optional[Path] = None,
):
    params.__post_init__()

    random.seed(seed)
    np.random.seed(seed)

    if not cfg_file.exists():
        raise FileNotFoundError(f"Missing cfg_file: {cfg_file}")

    pcr, voxel = load_cfg_ranges(cfg_file)
    split_root = data_root / split
    scenes = list_scenes(split_root)
    if not scenes:
        raise FileNotFoundError(f"No scenes found under: {split_root}")

    if scene_limit is not None:
        scenes = scenes.copy()
        random.shuffle(scenes)
        scenes = scenes[: int(scene_limit)]

    yolo = YOLO(params.MODEL_YOLO)

    sum_dist = 0.0
    sum_yaw_deg = 0.0
    cnt_pairs = 0
    cnt_pairs_with_yaw = 0
    sum_gt_in_bev = 0  # UNIQUE objects only

    for scene in scenes:
        scene_dir = split_root / scene
        stamps = list_stamps(scene_dir, agent="0")
        if not stamps:
            continue
        stamps_use = stamps[:: max(1, int(stamp_stride))]

        for ego_agent in ego_agents:
            for stamp in stamps_use:
                metas = load_metas_for_stamp(scene_dir, agents, stamp)
                if ego_agent not in metas:
                    continue

                pose_ego = get_sensor_pose_for_dets(metas[ego_agent])
                T_w_from_ego = T_world_from_pose_se2(pose_ego)
                T_ego_from_w = np.linalg.inv(T_w_from_ego)
                ego_yaw_world = math.radians(float(pose_ego[4]))

                occ_shape = load_occ_shape(bev_root, split, scene, ego_agent, stamp)

                dets_all = []

                for a, metaA in metas.items():
                    agent_dir = scene_dir / a
                    lidar_path = find_lidar_file(agent_dir, stamp)
                    if lidar_path is None:
                        continue
                    pts_lidar = load_lidar_bin(lidar_path)
                    if pts_lidar.shape[0] == 0:
                        continue

                    pose_a = get_sensor_pose_for_dets(metaA)
                    T_w_from_a = T_world_from_pose_se2(pose_a)
                    T_ego_from_a = np.linalg.inv(T_w_from_ego) @ T_w_from_a  # 3x3

                    for cam in cams:
                        img_path = agent_dir / f"{stamp}_{cam}.jpeg"
                        if (not img_path.exists()) or (cam not in metaA):
                            continue

                        img = cv2.imread(str(img_path))
                        if img is None:
                            continue
                        H, W = img.shape[:2]

                        raw_dets = None
                        cp = None
                        if yolo_cache_dir is not None:
                            cp = yolo_raw_cache_path(yolo_cache_dir, split, scene, a, stamp, cam)
                            raw_dets = load_json_silent(cp)
                        if raw_dets is None:
                            raw_dets = run_yolo_raw(yolo, img)
                            if yolo_cache_dir is not None and cp is not None:
                                save_json(cp, raw_dets)

                        dets = filter_raw_dets_for_params(raw_dets, params, agent=a, cam=cam, H=H, W=W)
                        if not dets:
                            continue

                        try:
                            K = np.array(metaA[cam]["intrinsic"], dtype=np.float64)
                            T_cam2lidar = to_4x4(metaA[cam]["extrinsic"])
                        except Exception:
                            continue

                        u, v, Zc, valid = project_lidar_to_image(
                            pts_lidar, K, T_cam2lidar, min_z_cam=params.MIN_Z_CAM_VIS
                        )
                        in_img = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)

                        for d in dets:
                            x1, y1, x2, y2 = d["xyxy"]

                            # center points
                            m_center = lidar_points_in_box(
                                u, v, Zc, in_img, x1, y1, x2, y2, H, W,
                                params=params,
                                use_bottom_strip=params.USE_BOTTOM_STRIP
                            )
                            idx_center = np.flatnonzero(m_center)
                            n_center = int(idx_center.size)
                            if n_center < params.MIN_LIDAR_PTS_IN_BOX:
                                continue

                            pts_a_xyz_c = pts_lidar[idx_center, :3]
                            Nc = pts_a_xyz_c.shape[0]
                            P_c = np.vstack([pts_a_xyz_c[:, 0], pts_a_xyz_c[:, 1], np.ones(Nc, dtype=np.float64)])
                            P_ce = T_ego_from_a @ P_c
                            pts_ego_xy_c = P_ce[:2, :].T

                            pts_center_xy = pts_ego_xy_c
                            if params.USE_CLUSTER_FOR_CENTER and pts_ego_xy_c.shape[0] >= params.CLUSTER_MIN_PTS:
                                clusters = cluster_xy_indices(
                                    pts_ego_xy_c, eps=params.CLUSTER_EPS_M, min_pts=params.CLUSTER_MIN_PTS
                                )
                                best = pick_best_cluster_indices(clusters)
                                if best is not None and len(best) >= params.CLUSTER_MIN_PTS:
                                    pts_center_xy = pts_ego_xy_c[np.array(best, dtype=int)]

                            x0 = float(np.median(pts_center_xy[:, 0]))
                            y0 = float(np.median(pts_center_xy[:, 1]))

                            # yaw points
                            m_yaw = lidar_points_in_box(
                                u, v, Zc, in_img, x1, y1, x2, y2, H, W,
                                params=params,
                                use_bottom_strip=(not params.YAW_USE_FULL_BOX)
                            )
                            idx_yaw = np.flatnonzero(m_yaw)

                            yaw_ego = None
                            if idx_yaw.size >= params.MIN_YAW_PTS:
                                pts_a_xyz_y = pts_lidar[idx_yaw, :3]
                                Ny = pts_a_xyz_y.shape[0]
                                P_y = np.vstack([pts_a_xyz_y[:, 0], pts_a_xyz_y[:, 1], np.ones(Ny, dtype=np.float64)])
                                P_ye = T_ego_from_a @ P_y
                                pts_ego_xy_y = P_ye[:2, :].T

                                pts_yaw_xy = pts_ego_xy_y
                                pts_yaw_a  = pts_a_xyz_y
                                if params.USE_CLUSTER_FOR_CENTER and pts_ego_xy_y.shape[0] >= params.CLUSTER_MIN_PTS:
                                    clusters2 = cluster_xy_indices(
                                        pts_ego_xy_y, eps=params.CLUSTER_EPS_M, min_pts=params.CLUSTER_MIN_PTS
                                    )
                                    best2 = pick_best_cluster_indices(clusters2)
                                    if best2 is not None and len(best2) >= params.CLUSTER_MIN_PTS:
                                        best2 = np.array(best2, dtype=int)
                                        pts_yaw_xy = pts_ego_xy_y[best2]
                                        pts_yaw_a  = pts_a_xyz_y[best2]

                                if params.TRIM_YAW_XY:
                                    keep = trim_yaw_points_xy_mask(pts_yaw_xy, d["cls"], params)
                                    if keep is not None:
                                        pts_yaw_xy = pts_yaw_xy[keep]
                                        pts_yaw_a  = pts_yaw_a[keep]

                                pts_yaw_xy_f = filter_ground_by_agent_z(pts_yaw_a, pts_yaw_xy, params)

                                y_pca = yaw_from_pca(pts_yaw_xy_f, params, last_yaw=None)
                                if y_pca is not None:
                                    yaw_ego = float(y_pca)

                                if yaw_ego is not None and params.USE_CLASS_PRIOR_90_DISAMBIG:
                                    yaw_ego = float(yaw_choose_by_class_prior(pts_yaw_xy_f, yaw_ego, d["cls"]))

                            if yaw_ego is not None:
                                yaw_ego = float(wrap_pi(yaw_ego + math.radians(params.YAW_OFFSET_DEG)))

                            dets_all.append({
                                "cls": d["cls"],
                                "conf": float(d["conf"]),
                                "x0": x0, "y0": y0,
                                "yaw_ego": yaw_ego,
                                "support": int(n_center),
                            })

                # Fuse by rotated overlap, then borrow yaw, then BEV filter + pairing
                fused = fuse_dets_better(dets_all, params)
                fused = borrow_yaw_from_neighbors(fused, max_dist=params.BORROW_YAW_MAX_DIST_M)

                fused_in = [d for d in fused if is_inside_bev_xy(
                    d["x0"], d["y0"], pcr, voxel,
                    occ_shape=occ_shape,
                    swap_xy=params.SWAP_XY_OCC,
                    margin_pix=1.0
                )]

                gt_list = collect_gt_ego(metas, T_ego_from_w, ego_yaw_world, params)
                gt_in = [g for g in gt_list if is_inside_bev_xy(
                    g["x0"], g["y0"], pcr, voxel,
                    occ_shape=occ_shape,
                    swap_xy=params.SWAP_XY_OCC,
                    margin_pix=1.0
                )]
                gt_unique = unique_gt_by_xy(gt_in, thr_m=params.GT_CLUSTER_THR_M)

                sum_gt_in_bev += len(gt_unique)

                pairs = pair_dets_to_gt(fused_in, gt_unique, gate_m=params.PAIR_GATE_M)
                if not pairs:
                    continue

                for (_, _, dist, yaw_err) in pairs:
                    sum_dist += float(dist)
                    cnt_pairs += 1
                    if yaw_err is not None:
                        sum_yaw_deg += math.degrees(float(yaw_err))
                        cnt_pairs_with_yaw += 1

    mean_dist = (sum_dist / cnt_pairs) if cnt_pairs > 0 else float("nan")
    mean_yaw = (sum_yaw_deg / cnt_pairs_with_yaw) if cnt_pairs_with_yaw > 0 else float("nan")
    yaw_valid_rate = (cnt_pairs_with_yaw / cnt_pairs) if cnt_pairs > 0 else 0.0
    coverage = (cnt_pairs / max(1, sum_gt_in_bev)) if sum_gt_in_bev > 0 else 0.0

    return {
        "split": split,
        "mean_dist_m": mean_dist,
        "mean_yaw_deg": mean_yaw,
        "matched_pairs": int(cnt_pairs),
        "pairs_with_yaw": int(cnt_pairs_with_yaw),
        "gt_in_bev": int(sum_gt_in_bev),
        "coverage": float(coverage),
        "yaw_valid_rate": float(yaw_valid_rate),
        "params": asdict(params),
    }


# =========================
# Optuna tuning
# =========================
def penalized_score(m: Dict[str, Any],
                    lambda_yaw=0.03,
                    pen_miss=10.0,
                    pen_yawmiss=3.0,
                    coverage_min=0.30,
                    yaw_rate_min=0.60):
    mean_dist = float(m.get("mean_dist_m", float("nan")))
    mean_yaw  = float(m.get("mean_yaw_deg", float("nan")))
    coverage  = float(m.get("coverage", 0.0))
    yaw_rate  = float(m.get("yaw_valid_rate", 0.0))

    if not (mean_dist == mean_dist):
        return 1e9
    if not (mean_yaw == mean_yaw):
        mean_yaw = 90.0

    miss_pen_hard = pen_miss * max(0.0, coverage_min - coverage) ** 2
    miss_pen_soft = pen_miss * 0.25 * (1.0 - coverage) ** 2
    yaw_pen_hard  = pen_yawmiss * max(0.0, yaw_rate_min - yaw_rate) ** 2
    yaw_pen_soft  = pen_yawmiss * 0.25 * (1.0 - yaw_rate) ** 2

    return float(mean_dist + lambda_yaw * mean_yaw + miss_pen_hard + miss_pen_soft + yaw_pen_hard + yaw_pen_soft)


def suggest_params(trial: optuna.Trial) -> Params:
    p = Params()

    # YOLO thresholds / NMS
    p.DET_CONF = trial.suggest_float("DET_CONF", 0.05, 0.60)
    p.DET_IOU  = trial.suggest_float("DET_IOU", 0.20, 0.80)

    p.USE_CLASS_AGNOSTIC_NMS = trial.suggest_categorical("USE_CLASS_AGNOSTIC_NMS", [True, False])
    if p.USE_CLASS_AGNOSTIC_NMS:
        p.NMS_IOU_AGNOSTIC = trial.suggest_float("NMS_IOU_AGNOSTIC", 0.40, 0.90)

    # Self-mask
    p.REMOVE_SELF_FROM_YOLO = trial.suggest_categorical("REMOVE_SELF_FROM_YOLO", [True, False])
    if p.REMOVE_SELF_FROM_YOLO:
        p.SELF_MASK_IOU_THR = trial.suggest_float("SELF_MASK_IOU_THR", 0.05, 0.35)

    # LiDAR-in-box
    p.USE_BOTTOM_STRIP = trial.suggest_categorical("USE_BOTTOM_STRIP", [True, False])
    if p.USE_BOTTOM_STRIP:
        p.BOTTOM_STRIP_FRAC = trial.suggest_float("BOTTOM_STRIP_FRAC", 0.20, 0.70)

    p.MIN_LIDAR_PTS_IN_BOX = trial.suggest_int("MIN_LIDAR_PTS_IN_BOX", 3, 30)
    p.MIN_RANGE_KEEP = trial.suggest_float("MIN_RANGE_KEEP", 0.5, 5.0)
    p.MAX_RANGE_KEEP = trial.suggest_float("MAX_RANGE_KEEP", 60.0, 150.0)

    # Center clustering
    p.USE_CLUSTER_FOR_CENTER = trial.suggest_categorical("USE_CLUSTER_FOR_CENTER", [True, False])
    if p.USE_CLUSTER_FOR_CENTER:
        p.CLUSTER_EPS_M = trial.suggest_float("CLUSTER_EPS_M", 0.2, 1.5)
        p.CLUSTER_MIN_PTS = trial.suggest_int("CLUSTER_MIN_PTS", 4, 25)

    # YAW (fixed method)
    p.YAW_USE_FULL_BOX = trial.suggest_categorical("YAW_USE_FULL_BOX", [True, False])
    p.USE_MINAREARECT_YAW = False
    p.USE_PCA_FALLBACK_YAW = True

    p.MIN_YAW_PTS = trial.suggest_int("MIN_YAW_PTS", 16, 40)
    p.GROUND_Z_PERCENTILE = trial.suggest_int("GROUND_Z_PERCENTILE", 10, 30)
    p.GROUND_Z_MARGIN = trial.suggest_float("GROUND_Z_MARGIN", 0.20, 0.50)
    p.SHAPE_EIG_RATIO_MIN = trial.suggest_float("SHAPE_EIG_RATIO_MIN", 1.6, 3.0)

    p.TRIM_YAW_XY = True
    p.TRIM_YAW_KEEP_SCALE = trial.suggest_float("TRIM_YAW_KEEP_SCALE", 0.6, 1.0)
    p.TRIM_YAW_SLACK_M = trial.suggest_float("TRIM_YAW_SLACK_M", 0.2, 1.0)
    p.TRIM_YAW_MIN_KEEP = trial.suggest_int("TRIM_YAW_MIN_KEEP", 8, 20)

    p.USE_CLASS_PRIOR_90_DISAMBIG = trial.suggest_categorical("USE_CLASS_PRIOR_90_DISAMBIG", [True, False])

    # overlap-based merge threshold
    p.MERGE_OVERLAP_MIN = trial.suggest_float("MERGE_OVERLAP_MIN", 0.05, 0.40)

    # pairing gate
    p.PAIR_GATE_M = trial.suggest_float("PAIR_GATE_M", 2.0, 10.0)

    # post-suppress
    p.FUSE_POST_SUPPRESS = trial.suggest_categorical("FUSE_POST_SUPPRESS", [True, False])
    if p.FUSE_POST_SUPPRESS:
        p.FUSE_SUPPRESS_DIST_M = trial.suggest_float("FUSE_SUPPRESS_DIST_M", 0.6, 2.5)
        p.FUSE_SUPPRESS_KEEP_BY = trial.suggest_categorical("FUSE_SUPPRESS_KEEP_BY", ["score", "support", "conf"])

    # yaw borrowing radius in meters
    p.BORROW_YAW_MAX_DIST_M = trial.suggest_float("BORROW_YAW_MAX_DIST_M", 1.0, 8.0)

    # fixed
    p.FUSE_WEIGHT_BY_SUPPORT = True
    p.YAW_OFFSET_DEG = 0.0
    p.SWAP_XY_OCC = False

    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--bev_root", type=Path, default=DEFAULT_BEV_ROOT)
    ap.add_argument("--cfg_file", type=Path, default=DEFAULT_CFG_FILE)

    ap.add_argument("--yolo_cache_dir", type=Path, default=DEFAULT_YOLO_CACHE_DIR)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)

    ap.add_argument("--yolo_model", type=str, default=None,
                    help="Override YOLO weights, e.g. yolov8x.pt or /path/to/custom.pt")

    ap.add_argument("--n_trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_scene_limit", type=int, default=8)
    ap.add_argument("--train_stamp_stride", type=int, default=5)
    ap.add_argument("--val_scene_limit", type=int, default=0, help="0 means None (all)")
    ap.add_argument("--val_stamp_stride", type=int, default=2)
    ap.add_argument("--topk_rerank_val", type=int, default=12)
    args, _ = ap.parse_known_args()

    DATA_ROOT = args.data_root
    BEV_ROOT  = args.bev_root
    CFG_FILE  = args.cfg_file
    YOLO_CACHE_DIR = args.yolo_cache_dir if args.yolo_cache_dir else None
    OUT_DIR = args.out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if YOLO_CACHE_DIR is not None:
        YOLO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if args.yolo_model is not None:
        # apply for all trials
        Params.MODEL_YOLO = args.yolo_model

    # Basic path sanity
    if not (DATA_ROOT / "train").is_dir():
        raise FileNotFoundError(f"Expected data_root/train to exist. Got data_root={DATA_ROOT}")
    if not CFG_FILE.exists():
        raise FileNotFoundError(f"Missing cfg_file: {CFG_FILE}")

    AGENTS = ["0", "1", "2", "3"]
    CAMS   = ["cam1", "cam2", "cam3", "cam4"]
    EGO_AGENTS = ("0", "1", "2", "3")

    TRAIN_SCENE_LIMIT = int(args.train_scene_limit)
    TRAIN_STAMP_STRIDE = int(args.train_stamp_stride)
    VAL_SCENE_LIMIT = None if int(args.val_scene_limit) == 0 else int(args.val_scene_limit)
    VAL_STAMP_STRIDE = int(args.val_stamp_stride)
    TOPK_RERANK_VAL = int(args.topk_rerank_val)

    sampler = optuna.samplers.TPESampler(seed=int(args.seed), multivariate=True, group=True)
    storage_path = OUT_DIR / "v2x_det_tuning.db"
    study = optuna.create_study(
        study_name="v2x_det_tuning",
        direction="minimize",
        sampler=sampler,
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=3),
    )

    def objective(trial: optuna.Trial) -> float:
        p = suggest_params(trial)

        m = evaluate_split(
            split="train",
            params=p,
            data_root=DATA_ROOT,
            bev_root=BEV_ROOT,
            cfg_file=CFG_FILE,
            agents=AGENTS,
            cams=CAMS,
            ego_agents=EGO_AGENTS,
            scene_limit=TRAIN_SCENE_LIMIT,
            stamp_stride=TRAIN_STAMP_STRIDE,
            seed=int(args.seed),
            yolo_cache_dir=YOLO_CACHE_DIR,
        )

        s = penalized_score(m)

        trial.set_user_attr("mean_dist_m", m.get("mean_dist_m"))
        trial.set_user_attr("mean_yaw_deg", m.get("mean_yaw_deg"))
        trial.set_user_attr("coverage", m.get("coverage"))
        trial.set_user_attr("yaw_valid_rate", m.get("yaw_valid_rate"))
        trial.set_user_attr("matched_pairs", m.get("matched_pairs"))
        trial.set_user_attr("gt_in_bev", m.get("gt_in_bev"))
        return s

    study.optimize(objective, n_trials=int(args.n_trials), gc_after_trial=True)

    trials_csv = OUT_DIR / "trials.csv"
    with open(trials_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "value", "mean_dist_m", "mean_yaw_deg", "coverage", "yaw_valid_rate",
                    "matched_pairs", "gt_in_bev", "params"])
        for t in study.trials:
            w.writerow([
                t.number, t.value,
                t.user_attrs.get("mean_dist_m"),
                t.user_attrs.get("mean_yaw_deg"),
                t.user_attrs.get("coverage"),
                t.user_attrs.get("yaw_valid_rate"),
                t.user_attrs.get("matched_pairs"),
                t.user_attrs.get("gt_in_bev"),
                dict(t.params),
            ])

    finished = [t for t in study.trials if t.value is not None]
    finished.sort(key=lambda t: t.value)
    top = finished[: int(TOPK_RERANK_VAL)]

    val_results = []
    for t in top:
        p = Params()
        for k, v in t.params.items():
            setattr(p, k, v)

        # enforce method
        p.USE_MINAREARECT_YAW = False
        p.USE_PCA_FALLBACK_YAW = True
        p.TRIM_YAW_XY = True
        p.__post_init__()

        mval = evaluate_split(
            split="val",
            params=p,
            data_root=DATA_ROOT,
            bev_root=BEV_ROOT,
            cfg_file=CFG_FILE,
            agents=AGENTS,
            cams=CAMS,
            ego_agents=EGO_AGENTS,
            scene_limit=VAL_SCENE_LIMIT,
            stamp_stride=VAL_STAMP_STRIDE,
            seed=123,
            yolo_cache_dir=YOLO_CACHE_DIR,
        )
        sval = penalized_score(mval)
        val_results.append((sval, t.number, p, mval))

    val_results.sort(key=lambda x: x[0])
    best_score, best_trial_num, best_params, best_metrics = val_results[0]

    best_yaml = OUT_DIR / "best_params.yaml"
    with open(best_yaml, "w") as f:
        bp = asdict(best_params)
        bp.pop("EGO_HOOD_MASK_FRAC", None)  # keep fixed in code
        yaml.safe_dump(bp, f, sort_keys=True)

    rerank_csv = OUT_DIR / "rerank_val.csv"
    with open(rerank_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "val_score", "trial_num", "mean_dist_m", "mean_yaw_deg", "coverage",
                    "yaw_valid_rate", "matched_pairs", "gt_in_bev"])
        for i, (sval, tnum, p, m) in enumerate(val_results, 1):
            w.writerow([i, sval, tnum,
                        m.get("mean_dist_m"), m.get("mean_yaw_deg"),
                        m.get("coverage"), m.get("yaw_valid_rate"),
                        m.get("matched_pairs"), m.get("gt_in_bev")])

    print("\n=== BEST (chosen on VAL) ===")
    print("best trial num:", best_trial_num)
    print("val score:", best_score)
    print("val metrics:", {k: best_metrics.get(k) for k in ["mean_dist_m","mean_yaw_deg","coverage","yaw_valid_rate","matched_pairs","gt_in_bev"]})
    print("saved:", best_yaml)
    print("trials:", trials_csv)
    print("rerank:", rerank_csv)
    print("study db:", storage_path)
    print("raw yolo cache:", YOLO_CACHE_DIR)


if __name__ == "__main__":
    main()

