# Cooperative-multi-Agent-V2X-Object-Detection-with-Vision-Based-Perception-
## Setup
pip install -r requirements.txt

## Run
python scripts/transform_lidar_to_ego.py \
  --data-root /path/to/data \
  --pc-root   /path/to/data \
  --out-root  /path/to/fused_points

python scripts/extract_bev_from_fused.py \
  --pcdet-root /path/to/OpenPCDet \
  --cfg-file   /path/to/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml \
  --ckpt-file  /path/to/OpenPCDet/checkpoints/pointpillar_18M.pth \
  --fused-root /path/to/fused_points \
  --save-root  /path/to/output_occ

python batch_eval_all_ego_uniqueGT_rotmerge.py \
  --split test \
  --data_root /path/to/data_root \
  --bev_root /path/to/BEV_root \
  --cfg_file /path/to/pointpillar.yaml \
  --best_params /path/to/best_params.yaml \
  --out_dir /path/to/output_dir
