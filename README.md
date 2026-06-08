<div align="center">

# Vision-Based Target-Specific Autonomous Vehicle Pursuit

An end-to-end CARLA research system for selecting one target vehicle, tracking it through traffic, estimating its relative pose, and controlling the ego vehicle to keep following that same target.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?logo=pytorch&logoColor=white)
![CARLA](https://img.shields.io/badge/CARLA-Simulator-0F172A)
![SAM3](https://img.shields.io/badge/SAM3-Target_Tracking-0F766E)
![LiDAR](https://img.shields.io/badge/LiDAR-3D_Perception-475569)
![CenterPoint](https://img.shields.io/badge/CenterPoint-3D_Detector-7C3AED)
![ConvNeXt](https://img.shields.io/badge/ConvNeXt-Target_Pose-2563EB)
![MPC](https://img.shields.io/badge/MPC-Follow_Controller-7C3AED)
![Dataset](https://img.shields.io/badge/Dataset-Multi--Town_CARLA-16A34A)

</div>

---

## Overview

TargetSpecificAVP solves a practical autonomous driving problem: **follow the selected car, not just whichever car is easiest to see**.

The system starts from a first-frame target prompt, uses SAM3 to maintain a target mask, predicts the selected vehicle's relative pose from RGB plus mask inputs, and feeds that pose into an MPC controller for closed-loop pursuit. The supporting ML stack generates a per-target CARLA dataset and trains the ConvNeXt-based pose regressor used by the controller.

The project has two main contributions:

1. **Target-specific AVP pipeline** - first-frame target selection, online mask tracking, learned relative pose estimation, and MPC control in CARLA.
2. **Dataset and training pipeline** - CARLA data collection with LiDAR plus 3D-detector pose supervision for ConvNeXt pose-regressor training.

Run the example commands below from the `TargetSpecificAVP/` directory.

## Highlights

| Capability | Detail |
|:-----------|:-------|
| Target identity persistence | **Follow one chosen car** instead of drifting to nearby traffic vehicles |
| Prompted initialization | **Bootstrap from a target bbox in the first frame**, then hand off to online SAM3 tracking |
| Target-specific perception | **Mask-conditioned pose estimation** predicts the selected car's `dx_m`, `dy_m`, and `yaw_follow_deg` |
| Closed-loop control | **MPC pursuit controller** uses the estimated target pose to keep following distance and alignment |
| Training setup | **ConvNeXt-based target pose learning** trained for the pursuit perception stack |
| Dataset engine | **Large and diverse CARLA dataset generation** with LiDAR and 3D-detector-derived pose labels |
| Alternate variant | Includes a **legacy stereo-depth perception variant** used in the current demo asset |

## Main Contributions

**1. End-to-end target-specific pursuit**

The runtime stack is built around target identity. A user or simulator prompt identifies the target vehicle in frame 0, SAM3 tracks that target online, the pose regressor estimates only the selected vehicle's relative pose, and MPC uses that estimate to keep the ego vehicle aligned with the target.

**2. Per-target dataset collection and labeling**

The data pipeline stores each accepted ego RGB frame once and creates one training sample per visible target actor. Each target sample has its own mask, bounding box, simulator-reference pose row, and detector-matched pose row.

**3. Learned target pose regressor**

The training pipeline learns a ConvNeXt-based regressor over full-frame RGB+mask, target-centered RGB+mask crop, and mask geometry features. It predicts `dx_m`, `dy_m`, and pursuit-aligned `yaw_follow_deg`.

**4. Closed-loop evaluation harness**

The inference harness runs the full SAM3 + CNN + MPC loop in CARLA and writes human-readable artifacts such as `metrics.json`, `frames.jsonl`, and `closed_loop_report.txt`.

## Demo

https://github.com/user-attachments/assets/87239f66-cd5d-4c5e-904d-cfed5f8a6532

This is a **legacy demo** from an earlier stereo-depth prototype. The pursuit target is the **blue car**. The current repo now focuses on the LiDAR-plus-3D-detector dataset pipeline and the target-specific SAM3 + CNN + MPC pursuit stack.

The local demo asset is available at [assets/RAVP_demo.mp4](assets/RAVP_demo.mp4).

## System Architecture

```text
First ego frame
  User or simulator provides target bbox
        |
        v
SAM3 online tracker
  Propagates the selected target mask
        |
        v
Target pose regressor
  Inputs: RGB frame, target mask, target crop, mask geometry
  Outputs: dx_m, dy_m, yaw_follow_deg
        |
        v
MPC pursuit controller
  Converts target pose into throttle, brake, and steer
        |
        v
Ego vehicle in CARLA traffic
  Continues following the designated target
```

The key interface is the target mask. It tells the perception model which vehicle matters when several cars are visible, allowing the controller to stay target-specific rather than lane- or nearest-car-specific.

## Dataset Collection Pipeline

The dataset pipeline in `carla_data_collection/` creates the supervised training data for target-relative pose estimation.

What it collects:

- shared ego RGB images in `rgb/`
- one binary target mask per accepted target actor in `masks/`
- simulator-reference labels in `gt_poses.csv`
- detector-matched labels in `pred_poses.csv`
- frame metadata in `frames.jsonl`
- collection totals and dataset diagnostics in `collection_summary.json`

Dataset layout:

```text
dataset/
├── rgb/
├── masks/
├── gt_poses.csv
├── pred_poses.csv
├── frames.jsonl
└── collection_summary.json
```

Important design choices:

- The dataset is **per-target**, not just per-frame.
- Multiple target vehicles can share the same RGB image while using different masks and pose rows.
- `gt_poses.csv` stores simulator-reference relative pose labels.
- `pred_poses.csv` stores detector-matched labels used by the default training setup.
- `dz_m` is kept for analysis, but the learned model predicts road-plane pursuit targets: `dx_m`, `dy_m`, and `yaw_follow_deg`.

Example collection command:

```bash
python -m carla_data_collection.run_collection collect-dataset \
    --output-dir ./RAVP_Dataset \
    --carla-host localhost \
    --carla-port 2150 \
    --towns Town01 Town02 Town03 Town04 Town05 \
    --follow-only \
    --min-follow-actors-per-frame 1 \
    --max-follow-actors-per-frame 4 \
    --target-samples-per-town 3000 \
    --max-frames-per-town 12000 \
    --num-traffic-vehicles 80 \
    --sam3-device cuda:0 \
    --detector-device cuda:0
```

Generate a detector-versus-reference report:

```bash
python -m carla_data_collection.run_collection report-metrics \
    --output-dir ./RAVP_Dataset
```

## Pose Regressor Training

The pose model in `target_pose_regression/` trains the perception module used by the pursuit controller.

Model inputs:

- full-frame `RGB + target_mask`
- target-centered crop `RGB + target_mask`
- mask geometry features

Model outputs:

- `dx_m`
- `dy_m`
- `yaw_follow_deg`

The default model is a shared-backbone ConvNeXt-Base regressor. Training uses grouped train/validation/test splits, translation normalization, weighted Smooth L1 translation loss, cosine yaw-vector loss, optional AMP, and optional Weights & Biases logging.

Example training command:

```bash
python -m target_pose_regression.train \
    --dataset-root ./RAVP_Dataset \
    --output-dir ./target_pose_runs \
    --label-source pred \
    --backbone convnext_base \
    --batch-size 16 \
    --num-epochs 40
```

Each saved run includes:

- `config.json`
- `translation_stats.json`
- `split_summary.json`
- `history.json`
- `best.pt`
- `last.pt`

## Closed-Loop Pursuit Inference

The inference stack in `inference/` runs the complete online pursuit loop:

1. Spawn a CARLA pursuit scenario.
2. Bootstrap the selected target from a first-frame bounding box.
3. Track the target with SAM3.
4. Estimate target-relative pose with the trained CNN.
5. Control the ego vehicle with MPC.
6. Save closed-loop metrics and optional debug videos.

Example pursuit command:

```bash
python -m inference.run_pursuit \
    --checkpoint-path ./target_pose_runs/<run>/best.pt \
    --town Town02 \
    --carla-host localhost \
    --carla-port 2150 \
    --sam3-device cuda:0 \
    --pose-device cuda:0 \
    --save-debug-images
```

Optional prompt control:

```bash
python -m inference.run_pursuit \
    --checkpoint-path ./target_pose_runs/<run>/best.pt \
    --bootstrap-bbox 120 180 420 620
```

Each inference run writes:

- `config.json`
- `metrics.json`
- `frames.jsonl`
- `closed_loop_report.txt`
- optional `ego.mp4`
- optional `spectator.mp4`
- optional tracker masks and debug frames

## Results / Artifacts

| Item | Status |
|:-----|:------:|
| Target-specific pursuit pipeline | Implemented |
| CARLA dataset generation | Implemented |
| Target pose regressor training | Implemented |
| Closed-loop metric logging | Implemented |
| Final benchmark table | In progress |

The repository contains the full collection, training, and pursuit components. Final benchmark numbers should be reported from saved training runs and closed-loop pursuit outputs.

## Project Structure

```text
TargetSpecificAVP/
├── assets/
│   └── RAVP_demo.mp4
├── carla_data_collection/
│   ├── collector.py
│   ├── config.py
│   ├── detector_3d.py
│   ├── ground_truth.py
│   ├── report_metrics.py
│   └── run_collection.py
├── inference/
│   ├── config.py
│   ├── metrics.py
│   ├── mpc_controller.py
│   ├── pose_estimator.py
│   ├── run_pursuit.py
│   ├── scenario.py
│   └── tracker.py
├── target_pose_regression/
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── preprocessing.py
│   └── train.py
├── scripts/
│   ├── start_carla_watchdog.sh
│   ├── start_collect_train_towns.sh
│   └── train_target_pose_models.sh
├── mmdet3d_models/
│   └── configs/
└── README.md
```

## Previous Prototype

The demo video comes from an earlier stereo-depth variant. That prototype used stereo images, a foundation stereo model, SAM3 masks, target point-cloud filtering, centroid translation estimates, and PCA-based yaw estimates to create pose labels.

That approach helped validate the target-specific pursuit idea, but the PCA-based yaw labels were noisy and produced weaker closed-loop behavior. The current repo moves the main training pipeline to CARLA LiDAR plus 3D-detector-matched labels, which gives a cleaner supervision path for the ConvNeXt pose regressor.

---

<div align="center">

Research Project at the **Computer Vision Group**, **Technical University of Munich**

</div>
