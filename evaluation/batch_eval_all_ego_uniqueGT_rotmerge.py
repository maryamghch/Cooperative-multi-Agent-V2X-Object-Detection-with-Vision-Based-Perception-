#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch evaluation over ALL scenes + ALL stamps + ALL ego agents (0..3), using the
SAME fusion logic as training:

Fusion: ROTATED BEV OVERLAP merge (intersection / min(area)) with MERGE_OVERLAP_MIN
After fuse: optional yaw borrowing within BORROW_YAW_MAX_DIST_M
Self-mask: applied ONLY for agents 0 and 1 (like training)
UNIQUE GT: deduplicate GT objects by XY clustering (count each object once)
Jupyter-safe parse_known_args()

Run (terminal):
  python batch_eval_all_ego_uniqueGT_rotmerge.py --split test

Run (Jupyter):
  just execute after saving; it will ignore the kernel's --f=... argument.
"""

from __future__ import annotations

import math
import csv
import argparse
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import yaml
import numpy as np
import cv2
from ultralytics import YOLO


# ================= USER DEFAULTS (portable) =================
from pathlib import Path

HERE = Path(__file__).resolve().parent  # folder where this .py lives

DEFAULT_SPLIT = "test"

# Put your repo/project root ONE level above this script if needed:
# e.g., scripts/this_file.py  -> project_root = HERE.parent
PROJECT_ROOT = HERE.parent

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_BEV_ROOT  = PROJECT_ROOT / "BEV_Features_Final_FUSED"
DEFAULT_CFG_FILE  = PROJECT_ROOT / "OpenPCDet" / "tools" / "cfgs" / "kitti_models" / "pointpillar.yaml"
DEFAULT_BEST_PARAMS_YAML = PROJECT_ROOT / "tuning_out" / "best_params.yaml"

AGENTS = ["0", "1", "2", "3"]
CAMS = ["cam1", "cam2", "cam3", "cam4"]

# =========================
# Class sizes (meters) for rotated BEV rectangles
# =========================
SIZE_BY_CLASS = {
    "car":   (4.5, 2.0),
    "van":   (5.2, 2.1),
    "truck": (8.0, 2.5),
    "bus":   (10.5, 2.6),
    "concretetruck": (9.5, 2.6),
}


# =========================
# PARAMS (match training)
# =========================
@dataclass
class Params:
    # YOLO
    MODEL_YOLO: str = "yolov8x.pt"
    YOLO_VEHICLE_CLASSES: Tuple[str, ...] = ("car", "van", "truck", "bus")

    DET_CONF: float = 0.30
    DET_IOU: float  = 0.40

    USE_CLASS_AGNOSTIC_NMS: bool = True
    NMS_IOU_AGNOSTIC: float = 0.60

    # Projection / filtering
    MIN_Z_CAM_VIS: float = 0.20

    USE_BOTTOM_STRIP: bool = False
    BOTTOM_STRIP_FRAC: float = 0.40
    MIN_LIDAR_PTS_IN_BOX: int = 20
    MIN_RANGE_KEEP: float = 0.0
    MAX_RANGE_KEEP: float = 150.0

    # Yaw (PCA-only, like training)
    YAW_USE_FULL_BOX: bool = True
    USE_MINAREARECT_YAW: bool = False
    USE_PCA_FALLBACK_YAW: bool = True
    MIN_YAW_PTS: int = 25
    SHAPE_EIG_RATIO_MIN: float = 2.0

    GROUND_Z_PERCENTILE: int = 15
    GROUND_Z_MARGIN: float = 0.25

    TRIM_YAW_XY: bool = True
    TRIM_YAW_KEEP_SCALE: float = 0.80
    TRIM_YAW_SLACK_M: float = 0.50
    TRIM_YAW_MIN_KEEP: int = 12

    USE_CLASS_PRIOR_90_DISAMBIG: bool = True
    YAW_OFFSET_DEG: float = 0.0

    # ===== Fusion (training-compatible) =====
    MERGE_OVERLAP_MIN: float = 0.10   # overlap (intersection / min(area))
    FUSE_POST_SUPPRESS: bool = True
    FUSE_SUPPRESS_DIST_M: float = 1.2
    FUSE_SUPPRESS_KEEP_BY: str = "score"  # "score" or "support" or "conf"
    FUSE_WEIGHT_BY_SUPPORT: bool = True

    BORROW_YAW_MAX_DIST_M: float = 3.0

    # ===== GT / pairing =====
    GT_CLUSTER_THR_M: float = 1.0
    VALID_GT_CLASSES: Tuple[str, ...] = ("car", "van", "truck", "bus", "concretetruck")
    PAIR_GATE_M: float = 2.5

    # ===== BEV bounds filter =====
    USE_BEV_BOUNDS_FILTER: bool = True
    BEV_MARGIN_PIX: float = 1.0
    SWAP_XY_OCC: bool = False

    # ===== Self detection removal (YOLO-stage mask) =====
    REMOVE_SELF_FROM_YOLO: bool = True
    SELF_MASK_IOU_THR: float = 0.20
    SELF_MASK_AGENTS: Tuple[str, ...] = ("0", "1")  # ✅ only for agents 0 and 1

    # Per-camera ego-hood mask as fractions of (W,H): (x1f,y1f,x2f,y2f)
    EGO_HOOD_MASK_FRAC: Optional[Dict[str, Tuple[float, float, float, float]]] = None

    # Metrics composite
    RECALL_PENALTY_M: float = 5.0

    # Optional: drop border-cut boxes (OFF by default to match training)
    REQUIRE_FULL_BOX_IN_IMAGE: bool = False
    BOX_EDGE_MARGIN_PX: int = 8
    MIN_BOX_W_PX: int = 20
    MIN_BOX_H_PX: int = 20

    def __post_init__(self):
        if self.EGO_HOOD_MASK_FRAC is None:
            self.EGO_HOOD_MASK_FRAC = {
                "cam1": (0.00, 0.78, 1.00, 1.00),
                "cam2": (0.00, 0.78, 1.00, 1.00),
                "cam3": (0.00, 0.78, 1.00, 1.00),
                "cam4": (0.00, 0.78, 1.00, 1.00),
            }


# ===================== Best params loader (apply into Params) =====================
def _coerce_to_field_type(params: Params, key: str, val: Any) -> Any:
    """
    Make YAML-loaded values safe:
    - cast floats/ints/bools to correct python types
    - handle "True"/"False" strings
    - keep strings for categorical options
    """
    # Find type from dataclass field definition
    ftype = None
    for f in fields(params):
        if f.name == key:
            ftype = f.type
            break

    # If unknown field, just return original
    if ftype is None:
        return val

    # Normalize numpy scalars
    if isinstance(val, (np.generic,)):
        val = val.item()

    # Handle bools that come as strings
    if isinstance(val, str):
        low = val.strip().lower()
        if low in ("true", "false"):
            val = (low == "true")

    # Cast based on current attribute type (most reliable here)
    cur = getattr(params, key, None)
    if isinstance(cur, bool):
        return bool(val)
    if isinstance(cur, int) and not isinstance(cur, bool):
        try:
            return int(val)
        except Exception:
            return cur
    if isinstance(cur, float):
        try:
            return float(val)
        except Exception:
            return cur
    if isinstance(cur, str):
        return str(val)

    # For tuples/dicts that we don't tune here, keep original
    return val


def apply_best_params(best_yaml: Path, params: Params) -> None:
    if not best_yaml.exists():
        raise FileNotFoundError(f"Best params YAML not found: {best_yaml}")

    d = yaml.safe_load(best_yaml.read_text()) or {}
    if not isinstance(d, dict):
        raise ValueError(f"{best_yaml} must contain a YAML dict of key:value pairs.")

    alias = {
        "USE_CLASS_AG": "USE_CLASS_AGNOSTIC_NMS",
        "USE_CLASS_AGNOSTIC_NMS": "USE_CLASS_AGNOSTIC_NMS",
    }

    ignored = []
    for k, v in d.items():
        kk = alias.get(k, k)
        if hasattr(params, kk):
            setattr(params, kk, _coerce_to_field_type(params, kk, v))
        else:
            ignored.append(k)

    if ignored:
        print(f"[WARN] Ignored {len(ignored)} keys from best_params.yaml not used in this script:")
        print("       " + ", ".join(map(str, ignored)))

    params.__post_init__()
    print(f"[OK] Applied tuned params from: {best_yaml}")
    print(f"     DET_CONF={params.DET_CONF}  DET_IOU={params.DET_IOU}  "
          f"PAIR_GATE_M={params.PAIR_GATE_M}  MERGE_OVERLAP_MIN={params.MERGE_OVERLAP_MIN}  "
          f"BORROW_YAW_MAX_DIST_M={params.BORROW_YAW_MAX_DIST_M}")


# ===================== Small helpers =====================
def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def ang_diff_axis(a: float, b: float) -> float:
    d = abs(wrap_pi(a - b))
    return min(d, math.pi - d)

def yaw_deg_pose(p):  # [x,y,z,roll,yaw,pitch]
    return float(p[4])

def normalize_gt_class(obj_type):
    if obj_type is None:
        return ""
    s = str(obj_type).strip().lower()
    s = s.replace("vehicle.", "").replace("carla.", "")
    s = s.replace(" ", "").replace("-", "").replace("_", "")
    if s in {"concretetruck", "cementtruck"}:
        return "concretetruck"
    return s

def to_4x4(m):
    m = np.array(m, dtype=np.float64)
    if m.shape == (4, 4):
        return m
    if m.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = m
        return out
    raise ValueError(f"Unexpected extrinsic shape: {m.shape}")

def clamp_xyxy(x1, y1, x2, y2, W, H):
    x1 = max(0.0, min(float(W - 1), float(x1)))
    x2 = max(0.0, min(float(W - 1), float(x2)))
    y1 = max(0.0, min(float(H - 1), float(y1)))
    y2 = max(0.0, min(float(H - 1), float(y2)))
    return x1, y1, x2, y2

def is_box_complete_in_image(x1, y1, x2, y2, W, H, margin_px=8, min_w=0, min_h=0):
    if (x2 - x1) < float(min_w) or (y2 - y1) < float(min_h):
        return False
    m = float(margin_px)
    return (x1 > m) and (y1 > m) and (x2 < (W - 1 - m)) and (y2 < (H - 1 - m))


# ===================== SE(2) =====================
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


# ===================== OpenPCDet cfg for BEV bounds =====================
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


# ===================== LiDAR IO =====================
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

def get_sensor_pose_for_dets(meta):
    # training code uses lidar_pose if available; otherwise true_ego_pose.
    if "lidar_pose" in meta and meta["lidar_pose"] is not None:
        return meta["lidar_pose"]
    return meta["true_ego_pose"]


# ===================== LiDAR projection (CAM->LiDAR) =====================
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


# ===================== YOLO + NMS (+ SELF MASK) =====================
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
    # ✅ only agents 0 and 1 masked
    if str(agent) not in set(params.SELF_MASK_AGENTS):
        return False
    m = ego_mask_xyxy(params, cam, H, W)
    if m is None:
        return False
    return box_iou_xyxy(box_xyxy, m) >= float(params.SELF_MASK_IOU_THR)

def run_yolo(params: Params, model: YOLO, img, agent: str, cam: str):
    H, W = img.shape[:2]
    r = model.predict(img, conf=float(params.DET_CONF), iou=float(params.DET_IOU), verbose=False)[0]

    allowed = set([c.lower() for c in params.YOLO_VEHICLE_CLASSES])
    out = []
    if r.boxes is None:
        return out

    names = model.names
    for b in r.boxes:
        cls = str(names[int(b.cls)]).lower()
        if cls not in allowed:
            continue

        x1, y1, x2, y2 = b.xyxy[0].tolist()
        x1, y1, x2, y2 = clamp_xyxy(x1, y1, x2, y2, W, H)
        box = (float(x1), float(y1), float(x2), float(y2))

        if params.REQUIRE_FULL_BOX_IN_IMAGE:
            if not is_box_complete_in_image(
                x1, y1, x2, y2, W, H,
                margin_px=params.BOX_EDGE_MARGIN_PX,
                min_w=params.MIN_BOX_W_PX,
                min_h=params.MIN_BOX_H_PX
            ):
                continue

        if params.REMOVE_SELF_FROM_YOLO and is_self_box_yolo(params, agent, cam, box, H, W):
            continue

        out.append({"cls": cls, "xyxy": box, "conf": float(b.conf)})

    # training does class-aware NMS first, then optional class-agnostic NMS
    if len(out) > 1:
        out = nms_class_aware(out, float(params.DET_IOU))
        if params.USE_CLASS_AGNOSTIC_NMS:
            out = nms_class_agnostic(out, float(params.NMS_IOU_AGNOSTIC))
    return out


# ===================== Ground filter + yaw (training-style) =====================
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

def yaw_from_pca_strict(pts_xy, params: Params, last_yaw=None):
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
        cands.append((errL + errW, y))
    cands.sort(key=lambda t: t[0])
    return cands[0][1]


# ===================== Rotated BEV overlap (training-style) =====================
def det_to_rotated_rect(d: dict) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """
    OpenCV RotatedRect: ((cx,cy), (w,h), angle_deg)
    BEV metric coordinates, angle in degrees CCW from +x.
    If yaw missing/non-finite, use 0.0.
    """
    cls = str(d.get("cls", "car"))
    Lm, Wm = SIZE_BY_CLASS.get(cls, (4.8, 2.2))
    yaw = d.get("yaw_ego", 0.0)
    if yaw is None or (not np.isfinite(yaw)):
        yaw = 0.0
    angle_deg = float(math.degrees(float(yaw)))
    return ((float(d["x0"]), float(d["y0"])), (float(Lm), float(Wm)), angle_deg)

def rotated_overlap_min_area(di: dict, dj: dict) -> float:
    """
    overlap = intersection_area / min(area_i, area_j)
    """
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


# ===================== Fusion (OVERLAP-BASED) =====================
def fuse_dets_overlap_rotated(dets: List[dict], params: Params) -> List[dict]:
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
        groups.setdefault(find(i), []).append(i)

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

    if (not params.FUSE_POST_SUPPRESS) or len(fused) <= 1:
        return fused

    key = str(params.FUSE_SUPPRESS_KEEP_BY)
    fused_sorted = sorted(fused, key=lambda d: -float(d.get(key, d.get("score", 0.0))))
    keep = []
    for d in fused_sorted:
        ok = True
        for k in keep:
            if math.hypot(d["x0"] - k["x0"], d["y0"] - k["y0"]) <= float(params.FUSE_SUPPRESS_DIST_M):
                ok = False
                break
        if ok:
            keep.append(d)
    return keep


# ===================== Yaw borrowing (training-style) =====================
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


# ===================== UNIQUE GT (cluster XY; count each object once) =====================
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
            yaw_ego = wrap_pi(yaw_w - ego_yaw_world + math.radians(params.YAW_OFFSET_DEG))

            gt_list.append({"x0": x0, "y0": y0, "yaw_ego": yaw_ego, "cls": cls})
    return gt_list

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


# ===================== Pairing + metrics =====================
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
            pairs.append((d, g, float(best_d), yaw_err))
    return pairs

def compute_metrics(fused_in, gt_in, pairs, recall_penalty_m=5.0):
    n_det = int(len(fused_in))
    n_gt  = int(len(gt_in))
    n_match = int(len(pairs))

    dists = np.array([p_[2] for p_ in pairs], dtype=np.float64) if n_match > 0 else np.array([], dtype=np.float64)
    yaw_deg = np.array([math.degrees(p_[3]) for p_ in pairs if p_[3] is not None], dtype=np.float64)

    recall = (n_match / n_gt) if n_gt > 0 else (1.0 if n_det == 0 else 0.0)
    precision = (n_match / n_det) if n_det > 0 else (1.0 if n_gt == 0 else 0.0)

    out = {
        "n_det": n_det,
        "n_gt": n_gt,
        "n_match": n_match,
        "recall": float(recall),
        "precision": float(precision),

        "mean_dist_m": float(np.mean(dists)) if dists.size else float("nan"),
        "median_dist_m": float(np.median(dists)) if dists.size else float("nan"),
        "p90_dist_m": float(np.percentile(dists, 90)) if dists.size else float("nan"),
        "max_dist_m": float(np.max(dists)) if dists.size else float("nan"),

        "mean_yaw_axis_deg": float(np.mean(yaw_deg)) if yaw_deg.size else float("nan"),
        "median_yaw_axis_deg": float(np.median(yaw_deg)) if yaw_deg.size else float("nan"),
        "max_yaw_axis_deg": float(np.max(yaw_deg)) if yaw_deg.size else float("nan"),
    }

    med = out["median_dist_m"]
    if not np.isfinite(med):
        med = 0.0
    out["score_med_plus_recallpen_m"] = float(med + float(recall_penalty_m) * (1.0 - float(recall)))
    return out


# ===================== Scene/stamp discovery =====================
def list_scenes(split_dir: Path):
    if not split_dir.exists():
        return []
    scenes = [p.name for p in split_dir.iterdir() if p.is_dir()]
    scenes.sort()
    return scenes

def list_stamps_for_scene(scene_dir: Path, ego_agent: str):
    agent_dir = scene_dir / ego_agent
    if not agent_dir.exists():
        return []
    stamps = []
    for yp in agent_dir.glob("*.yaml"):
        stem = yp.stem
        if stem.isdigit():  # ✅ avoid accidental non-stamp yamls
            stamps.append(stem)
    stamps.sort()
    return stamps


# ===================== Evaluate one stamp for one ego =====================
def eval_one_stamp_one_ego(
    params: Params,
    data_root_split: Path,
    bev_root_split: Path,
    cfg_file: Path,
    scene: str,
    stamp: str,
    ego_agent: str,
    yolo: YOLO,
    pcr: np.ndarray,
    voxel: np.ndarray,
):
    scene_dir = data_root_split / scene
    if not scene_dir.exists():
        return None

    # load metas for this stamp for all agents that exist
    metas = {}
    for a in AGENTS:
        yp = scene_dir / a / f"{stamp}.yaml"
        if yp.exists():
            metas[a] = yaml.safe_load(open(yp, "r"))
    if ego_agent not in metas:
        return None

    pose_ego = get_sensor_pose_for_dets(metas[ego_agent])
    T_w_from_ego = T_world_from_pose_se2(pose_ego)
    T_ego_from_w = np.linalg.inv(T_w_from_ego)
    ego_yaw_world = math.radians(float(pose_ego[4]))

    # Optional BEV occ shape for bounds
    occ_shape = None
    occ_path = bev_root_split / scene / ego_agent / f"{stamp}_occ.npy"
    if occ_path.exists():
        try:
            occ = np.load(occ_path, mmap_mode="r")
            occ_shape = occ.shape
        except Exception:
            occ_shape = None

    dets_all = []

    # Loop agents/cams to build dets in THIS ego frame
    for a, metaA in metas.items():
        agent_dir = scene_dir / a
        lidar_path = find_lidar_file(agent_dir, stamp)
        if lidar_path is None:
            continue

        pts_lidar = load_lidar_bin(lidar_path)
        if pts_lidar.shape[0] == 0:
            continue

        # agent->ego (SE2)
        pose_a = get_sensor_pose_for_dets(metaA)
        T_w_from_a = T_world_from_pose_se2(pose_a)
        T_ego_from_a = np.linalg.inv(T_w_from_ego) @ T_w_from_a  # 3x3

        for cam in CAMS:
            img_path = agent_dir / f"{stamp}_{cam}.jpeg"
            if (not img_path.exists()) or (cam not in metaA):
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            H, W = img.shape[:2]

            try:
                K = np.array(metaA[cam]["intrinsic"], dtype=np.float64)
                T_cam2lidar = to_4x4(metaA[cam]["extrinsic"])
            except Exception:
                continue

            u, v, Zc, valid = project_lidar_to_image(pts_lidar, K, T_cam2lidar, min_z_cam=params.MIN_Z_CAM_VIS)
            in_img = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)

            dets = run_yolo(params, yolo, img, agent=a, cam=cam)
            if not dets:
                continue

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
                if n_center < int(params.MIN_LIDAR_PTS_IN_BOX):
                    continue

                pts_a_xyz_c = pts_lidar[idx_center, :3]
                Nc = pts_a_xyz_c.shape[0]
                P_c = np.vstack([pts_a_xyz_c[:, 0], pts_a_xyz_c[:, 1], np.ones(Nc, dtype=np.float64)])
                P_ce = T_ego_from_a @ P_c
                pts_ego_xy_c = P_ce[:2, :].T

                x0 = float(np.median(pts_ego_xy_c[:, 0]))
                y0 = float(np.median(pts_ego_xy_c[:, 1]))

                # yaw points (full box or bottom strip based on params)
                m_yaw = lidar_points_in_box(
                    u, v, Zc, in_img, x1, y1, x2, y2, H, W,
                    params=params,
                    use_bottom_strip=(not params.YAW_USE_FULL_BOX)
                )
                idx_yaw = np.flatnonzero(m_yaw)

                yaw_ego = None
                if idx_yaw.size >= int(params.MIN_YAW_PTS):
                    pts_a_xyz_y = pts_lidar[idx_yaw, :3]
                    Ny = pts_a_xyz_y.shape[0]
                    P_y = np.vstack([pts_a_xyz_y[:, 0], pts_a_xyz_y[:, 1], np.ones(Ny, dtype=np.float64)])
                    P_ye = T_ego_from_a @ P_y
                    pts_ego_xy_y = P_ye[:2, :].T

                    # ground filter + optional XY trim (training-style)
                    pts_yaw_xy = filter_ground_by_agent_z(pts_a_xyz_y, pts_ego_xy_y, params)
                    if params.TRIM_YAW_XY:
                        keep = trim_yaw_points_xy_mask(pts_yaw_xy, d["cls"], params)
                        if keep is not None:
                            pts_yaw_xy = pts_yaw_xy[keep]

                    y_pca = yaw_from_pca_strict(pts_yaw_xy, params, last_yaw=None)
                    if y_pca is not None:
                        yaw_ego = float(y_pca)
                        if params.USE_CLASS_PRIOR_90_DISAMBIG:
                            yaw_ego = float(yaw_choose_by_class_prior(pts_yaw_xy, yaw_ego, d["cls"]))

                if yaw_ego is not None:
                    yaw_ego = float(wrap_pi(yaw_ego + math.radians(params.YAW_OFFSET_DEG)))

                dets_all.append({
                    "cls": d["cls"],
                    "conf": float(d["conf"]),
                    "x0": x0, "y0": y0,
                    "yaw_ego": yaw_ego,
                    "support": int(n_center),
                    "agent": a, "cam": cam,
                })

    # Fuse by rotated overlap, then borrow yaw
    fused = fuse_dets_overlap_rotated(dets_all, params)
    fused = borrow_yaw_from_neighbors(fused, max_dist=float(params.BORROW_YAW_MAX_DIST_M))

    # UNIQUE GT in THIS ego frame
    gt_raw = collect_gt_ego(metas, T_ego_from_w, ego_yaw_world, params)
    gt_unique = unique_gt_by_xy(gt_raw, thr_m=float(params.GT_CLUSTER_THR_M))

    # optional BEV bounds filter
    if params.USE_BEV_BOUNDS_FILTER and (occ_shape is not None):
        fused_in = [d for d in fused if is_inside_bev_xy(
            d["x0"], d["y0"], pcr, voxel,
            occ_shape=occ_shape, swap_xy=params.SWAP_XY_OCC, margin_pix=float(params.BEV_MARGIN_PIX)
        )]
        gt_in = [g for g in gt_unique if is_inside_bev_xy(
            g["x0"], g["y0"], pcr, voxel,
            occ_shape=occ_shape, swap_xy=params.SWAP_XY_OCC, margin_pix=float(params.BEV_MARGIN_PIX)
        )]
        use_bev = 1
    else:
        fused_in = fused
        gt_in = gt_unique
        use_bev = 0

    pairs = pair_dets_to_gt(fused_in, gt_in, gate_m=float(params.PAIR_GATE_M))
    metrics = compute_metrics(fused_in, gt_in, pairs, recall_penalty_m=float(params.RECALL_PENALTY_M))

    metrics.update({
        "scene": scene,
        "stamp": stamp,
        "ego_agent": str(ego_agent),
        "use_bev_filter": int(use_bev),
        "n_fused_raw": int(len(fused)),
        "n_gt_unique_raw": int(len(gt_unique)),
        "n_pairs": int(len(pairs)),
    })
    return metrics


# ===================== Aggregation utils =====================
def _safe_nanmean(vals):
    a = np.array(vals, dtype=np.float64)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")

def summarize_rows(rows):
    if not rows:
        return None

    total_det = sum(int(r["n_det"]) for r in rows)
    total_gt = sum(int(r["n_gt"]) for r in rows)
    total_match = sum(int(r["n_match"]) for r in rows)

    w = np.array([max(0, int(r["n_match"])) for r in rows], dtype=np.float64)

    def wmean(key):
        v = np.array([r.get(key, float("nan")) for r in rows], dtype=np.float64)
        m = np.isfinite(v) & (w > 0)
        if not m.any():
            return float("nan")
        return float(np.sum(v[m] * w[m]) / np.sum(w[m]))

    out = {
        "n_rows": int(len(rows)),
        "total_det": int(total_det),
        "total_gt": int(total_gt),
        "total_match": int(total_match),
        "overall_recall": float(total_match / total_gt) if total_gt > 0 else float("nan"),
        "overall_precision": float(total_match / total_det) if total_det > 0 else float("nan"),
        "mean_dist_m_wmatch": wmean("mean_dist_m"),
        "mean_yaw_axis_deg_wmatch": wmean("mean_yaw_axis_deg"),
        "median_dist_m_perrow_mean": _safe_nanmean([r.get("median_dist_m", float("nan")) for r in rows]),
        "score_perrow_mean": _safe_nanmean([r.get("score_med_plus_recallpen_m", float("nan")) for r in rows]),
        "recall_perrow_mean": _safe_nanmean([r.get("recall", float("nan")) for r in rows]),
        "precision_perrow_mean": _safe_nanmean([r.get("precision", float("nan")) for r in rows]),
    }
    return out


# ===================== CLI =====================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default=DEFAULT_SPLIT, choices=["train", "val", "test"])
    ap.add_argument("--data_root", type=str, default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--bev_root", type=str, default=str(DEFAULT_BEV_ROOT))
    ap.add_argument("--cfg_file", type=str, default=str(DEFAULT_CFG_FILE))
    ap.add_argument("--best_params", type=str, default=str(DEFAULT_BEST_PARAMS_YAML))
    ap.add_argument("--out_dir", type=str, default="/home/yoda/Desktop/Maryam/New/batch_eval_out_all_ego")
    ap.add_argument("--limit_scenes", type=int, default=0, help="0 = no limit")
    ap.add_argument("--limit_stamps", type=int, default=0, help="0 = no limit")
    args, _unknown = ap.parse_known_args()  # ✅ Jupyter-safe
    return args


def main():
    args = parse_args()

    split = args.split
    data_root_split = Path(args.data_root) / split
    bev_root_split  = Path(args.bev_root) / split
    cfg_file = Path(args.cfg_file)
    best_yaml = Path(args.best_params)

    out_dir = Path(args.out_dir) / split
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "per_stamp_metrics.csv"
    out_sum = out_dir / "summary_overall.txt"
    out_per_ego = out_dir / "summary_per_ego.csv"

    params = Params()
    apply_best_params(best_yaml, params)

    pcr, voxel = load_cfg_ranges(cfg_file)
    yolo = YOLO(params.MODEL_YOLO)

    scenes = list_scenes(data_root_split)
    if args.limit_scenes and args.limit_scenes > 0:
        scenes = scenes[:args.limit_scenes]

    rows = []
    n_done = 0

    for scene in scenes:
        scene_dir = data_root_split / scene
        stamps = list_stamps_for_scene(scene_dir, ego_agent="0")
        if not stamps:
            continue
        if args.limit_stamps and args.limit_stamps > 0:
            stamps = stamps[:args.limit_stamps]

        for stamp in stamps:
            for ego in AGENTS:
                m = eval_one_stamp_one_ego(
                    params=params,
                    data_root_split=data_root_split,
                    bev_root_split=bev_root_split,
                    cfg_file=cfg_file,
                    scene=scene,
                    stamp=stamp,
                    ego_agent=ego,
                    yolo=yolo,
                    pcr=pcr,
                    voxel=voxel,
                )
                if m is None:
                    continue
                rows.append(m)
                n_done += 1
                if n_done % 100 == 0:
                    print(f"[PROGRESS] processed {n_done} rows... latest={scene}/{stamp} ego={ego}")

    if not rows:
        print("[DONE] No rows evaluated (check paths / split / file structure).")
        return

    # Write CSV
    fieldnames = [
        "scene","stamp","ego_agent","use_bev_filter",
        "n_det","n_gt","n_match","recall","precision",
        "mean_dist_m","median_dist_m","p90_dist_m","max_dist_m",
        "mean_yaw_axis_deg","median_yaw_axis_deg","max_yaw_axis_deg",
        "score_med_plus_recallpen_m",
        "n_fused_raw","n_gt_unique_raw","n_pairs",
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"[OK] wrote per-stamp CSV: {out_csv}")

    # Overall summary
    overall = summarize_rows(rows)
    with open(out_sum, "w") as f:
        f.write("BATCH EVAL SUMMARY (UNIQUE GT) - ALL EGO AGENTS\n")
        f.write(f"split={split}\n")
        f.write(f"rows(scene,stamp,ego)={overall['n_rows']}\n\n")
        f.write(f"TOTAL det={overall['total_det']}  gt={overall['total_gt']}  match={overall['total_match']}\n")
        f.write(f"OVERALL recall={overall['overall_recall']:.4f}  precision={overall['overall_precision']:.4f}\n\n")
        f.write("ERRORS (matched-only, weighted by matches):\n")
        f.write(f"  mean_dist_m = {overall['mean_dist_m_wmatch']:.4f}\n")
        f.write(f"  mean_yaw_axis_deg = {overall['mean_yaw_axis_deg_wmatch']:.4f}\n\n")
        f.write("PER-ROW AVERAGES:\n")
        f.write(f"  mean(median_dist_m) = {overall['median_dist_m_perrow_mean']:.4f}\n")
        f.write(f"  mean(recall) = {overall['recall_perrow_mean']:.4f}\n")
        f.write(f"  mean(precision) = {overall['precision_perrow_mean']:.4f}\n")
        f.write(f"  mean(score = median_dist + {params.RECALL_PENALTY_M}*(1-recall)) = {overall['score_perrow_mean']:.4f}\n")
    print(f"[OK] wrote overall summary: {out_sum}")

    # Per-ego summary CSV
    by_ego = {}
    for r in rows:
        by_ego.setdefault(str(r["ego_agent"]), []).append(r)

    per_ego_fieldnames = [
        "ego_agent","n_rows",
        "total_det","total_gt","total_match",
        "overall_recall","overall_precision",
        "mean_dist_m_wmatch","mean_yaw_axis_deg_wmatch",
        "mean_recall_perrow","mean_precision_perrow",
        "mean_score_perrow",
    ]

    with open(out_per_ego, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=per_ego_fieldnames)
        w.writeheader()
        for ego in AGENTS:
            s = summarize_rows(by_ego.get(ego, []))
            if s is None:
                continue
            w.writerow({
                "ego_agent": ego,
                "n_rows": s["n_rows"],
                "total_det": s["total_det"],
                "total_gt": s["total_gt"],
                "total_match": s["total_match"],
                "overall_recall": s["overall_recall"],
                "overall_precision": s["overall_precision"],
                "mean_dist_m_wmatch": s["mean_dist_m_wmatch"],
                "mean_yaw_axis_deg_wmatch": s["mean_yaw_axis_deg_wmatch"],
                "mean_recall_perrow": s["recall_perrow_mean"],
                "mean_precision_perrow": s["precision_perrow_mean"],
                "mean_score_perrow": s["score_perrow_mean"],
            })
    print(f"[OK] wrote per-ego summary CSV: {out_per_ego}")
    print(f"[DONE] outputs at: {out_dir}")


if __name__ == "__main__":
    main()
