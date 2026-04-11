<div align="center">

# Vision-Based Target-Specific Autonomous Vehicle Pursuit

### Robust target-specific vehicle pursuit in CARLA

Track and follow one designated target vehicle through traffic using a first-frame target prompt, SAM3 tracking, learned relative pose estimation, and MPC-based control of the ego vehicle.

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

## Highlights

| Capability | Detail |
|:-----------|:-------|
| Target identity persistence | **Follow one chosen car** instead of drifting to nearby traffic vehicles|
| Prompted initialization | **Bootstrap from a target bbox in the first frame**, then hand off to online SAM3 tracking |
| Target-specific perception | **Mask-conditioned pose estimation** predicts the selected car's `dx_m`, `dy_m`, and `yaw_follow_deg` |
| Closed-loop control | **MPC pursuit controller** uses the estimated target pose to keep following distance and alignment |
| Training setup | **ConvNeXt-based target pose learning** trained for the pursuit perception stack |
| Dataset engine | **Large and diverse CARLA dataset generation** with per-target supervision, LiDAR, and detector-derived labels |
| Alternate variant | Includes a **legacy stereo-depth perception variant** used in the current demo asset |

---

## Demo

https://github.com/user-attachments/assets/87239f66-cd5d-4c5e-904d-cfed5f8a6532

This is a **legacy demo**. The pursuit target is the **blue car**.

The current file in [assets/RAVP_demo.mp4](/my_workspace/Resume/RAVP/assets/RAVP_demo.mp4) comes from the legacy stereo-depth variant, where the RGB-plus-mask CNN was trained using stereo-and-PCA-derived pose labels instead of the current LiDAR-plus-3D-detector label-generation pipeline.

---

## What It Does

RAVP is about **following the correct car**, not just following any car ahead. The core problem is target-specific vehicle pursuit in traffic: if multiple vehicles are visible in front of the ego car, the system should stay locked onto the designated target instead of switching to a distractor.

The intended deployment flow is:

1. The user provides the **bounding box of the target vehicle in the first frame**
2. **SAM3** tracks that target online and produces a target mask for each new ego-view frame
3. The target mask and RGB image are fed to a **CNN pose regressor** that predicts the selected vehicle's relative pose
4. An **MPC controller** uses that pose estimate to keep following the same target vehicle

This makes the system target-aware: the pursuit policy is conditioned on the tracked target mask, so two similar vehicles ahead do not look identical to the controller.

For controlled CARLA evaluation, the current `pursuit_eval` harness bootstraps frame 0 using a simulator-projected target box by default. Conceptually, this is the same interface as a user-provided first-frame target prompt.

The repo has two clearly separated layers:

- The **pursuit stack** is the main story: target prompt, SAM3 tracking, pose estimation, and MPC control
- The **data and training stack** exists to support that pursuit stack with supervised learning and evaluation data

---

## Architecture

```text
        ┌─────────────────────────────────────┐
        │  First Ego Frame                    │
        │  User selects target bbox           │
        └──────────────────┬──────────────────┘
                           │ Prompted target initialization
                           ▼
        ┌─────────────────────────────────────┐
        │  SAM3 Online Tracker                │
        │  Per-frame target mask propagation  │
        └──────────────────┬──────────────────┘
                           │ RGB frame + target mask
                           ▼
        ┌─────────────────────────────────────┐
        │  Target Pose Regressor              │
        │  Shared ConvNeXt global/local views │
        └──────────────────┬──────────────────┘
                           │ dx_m, dy_m, yaw_follow_deg
                           ▼
        ┌─────────────────────────────────────┐
        │  MPC Pursuit Controller             │
        │  Target-specific follow control     │
        └──────────────────┬──────────────────┘
                           │ throttle / brake / steer
                           ▼
        ┌─────────────────────────────────────┐
        │  Ego Vehicle in Traffic             │
        │  Continues following chosen target  │
        └─────────────────────────────────────┘
```

Supporting training pipeline:

```text
CARLA Episodes
    → per-target dataset generation
    → pose labels and masks
    → target_pose_regression training
    → checkpoint used in pursuit inference
```

---

## Current And Legacy Variants

### Current main pipeline

The main branch of the project is centered on:

- first-frame target initialization
- SAM3 target-mask tracking
- a learned `RGB + target_mask` pose regressor
- MPC-based target following

The learned model predicts:

- `dx_m`
- `dy_m`
- `yaw_follow_deg`

`yaw_follow_deg` is a pursuit-aligned folded yaw target. It keeps the forward-facing equivalent heading in `[-90, 90]`, which is more useful for follow control than raw relative yaw.

### Legacy stereo-depth variant used in the demo asset

The current demo video asset is a **legacy demo** from an earlier variant where the main difference was the **data-generation / pose-labeling strategy**, not the high-level pursuit objective.

That legacy branch still trained a CNN from **RGB + target mask** for target-relative pose prediction. The key difference was how the pose supervision was generated.

That older variant worked as follows:

1. Collect **stereo images** in CARLA
2. Use a **foundation stereo model** to estimate metric depth
3. Convert the depth map into a **depth point cloud**
4. Use **SAM3** on RGB to instance-segment cars
5. Use the target mask to filter the depth point cloud and isolate the target car point cloud
6. Estimate `dx` and `dy` from the **centroid** of the target point cloud
7. Estimate `dyaw` from the **first PCA component** of the target car point cloud, assuming it roughly aligns with the vehicle's longitudinal axis

So the legacy pipeline still used an RGB-plus-mask CNN, but its training labels came from stereo depth, filtered target point clouds, centroid translation estimates, and PCA-based yaw estimates instead of the current LiDAR-plus-3D-detector pipeline.

In practice, the `dx`, `dy`, and `dyaw` errors from that PCA-based pose-labeling path were fairly high. Those noisy labels hurt downstream CNN training, which then produced weaker pursuit behavior with noticeable **oscillatory lateral corrections** rather than smooth target following.

So the demo still reflects the same target-pursuit idea, but it is not showcasing the current version of the RGB-plus-mask CNN pipeline, which uses the newer LiDAR-plus-3D-detector supervision path.

---

## Results

| Item | Status |
|:-----|:------:|
| Target-specific pursuit pipeline | **Implemented** |
| CARLA dataset generation | **Implemented** |
| Target pose training | **Implemented** |
| End-to-end benchmark table | **In progress** |

The repository already contains the full collection, training, and pursuit components. Final quantitative benchmark numbers should be reported from the saved training runs and pursuit-evaluation outputs.

---

## Pursuit Pipeline

The runtime story of the project is:

| Stage | Role |
|:------|:-----|
| **Target prompt** | Choose which car the ego vehicle should follow |
| **Online tracking** | Use SAM3 to maintain a target mask over time |
| **Pose estimation** | Predict the chosen target's relative pose from `RGB + mask` |
| **Control** | Feed the relative pose into MPC and generate control commands |

The most important behavior is **identity consistency**. The target mask tells the model which vehicle matters, so the ego vehicle can continue following the designated target even when other nearby vehicles are visually similar or occupy similar lanes.

To keep the project easy to reason about:

- `pursuit_eval/` and `inference/` are the runtime pursuit side
- `carla_data_collection/` and `target_pose_regression/` are the supporting dataset-generation and learning side

The current CARLA evaluation harness lives in `pursuit_eval/` and supports detector-driven online pursuit with in-process SAM3 tracking and 3D detection.

Example pursuit evaluation command:

```bash
python -m pursuit_eval.run \
    --pose-source detector \
    --town Town02 \
    --carla-host localhost \
    --carla-port 2150 \
    --sam3-device cuda:0 \
    --detector-device cuda:0
```

---

## Pose Model

The learned pose model lives in `target_pose_regression/`.

It uses:

- a full-frame `RGB + target_mask` view
- a target-centered crop `RGB + target_mask` view
- mask geometry features

The default model is a shared-backbone **ConvNeXt-Base** regressor. It predicts the target vehicle's relative translation and pursuit-aligned yaw:

- `dx_m`
- `dy_m`
- `yaw_follow_deg`

The current pursuit-focused training setup uses detector-matched pose supervision from `pred_poses.csv`.

Example training commands:

```bash
python -m target_pose_regression.train \
    --dataset-root /my_workspace/Resume/RAVP_Dataset \
    --output-dir /my_workspace/Resume/RAVP/target_pose_runs
```

The saved run directory includes:

- `config.json`
- `translation_stats.json`
- `split_summary.json`
- `history.json`
- `best.pt`
- `last.pt`

W&B logging is enabled by default for losses only.

---

## Dataset

RAVP uses a large and diverse CARLA dataset built for target-specific pursuit. The dataset is generated directly from CARLA as a per-target dataset: each accepted ego frame is stored once in `rgb/`, and every accepted target actor contributes its own binary mask and pose row.

What the dataset is:

- a multi-town CARLA driving dataset for target-relative vehicle pose learning
- each sample corresponds to one chosen target vehicle in one ego-view frame
- supervision includes simulator-reference pose labels and detector-matched pose labels
- the data-generation pipeline uses ego RGB, instance segmentation, LiDAR, SAM3 masks, and a 3D detector
- the goal of the dataset is to train and evaluate the pursuit perception module, not just to archive frames

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

Important semantics:

- The dataset is **per-target**, not per-frame
- Multiple target cars in one ego frame share the same RGB image but use different masks
- `gt_poses.csv` stores simulator-reference relative pose labels
- `pred_poses.csv` stores detector-matched pose labels for the same accepted targets
- `dz_m` is kept in the CSVs for analysis, but the learned target excludes it because the target vehicle is assumed to stay on the road plane

The dataset-generation pipeline lives in `carla_data_collection/` and writes the final dataset directly in one pass instead of saving large raw-replay caches.

Example collection command:

```bash
python -m carla_data_collection.run_collection collect-dataset \
    --output-dir /my_workspace/Resume/RAVP_Dataset \
    --carla-host localhost \
    --carla-port 2150 \
    --towns Town01 Town01_Opt Town02 Town02_Opt Town03 Town03_Opt Town04 Town04_Opt Town05 Town05_Opt \
    --follow-only \
    --min-follow-actors-per-frame 1 \
    --max-follow-actors-per-frame 4 \
    --follow-lateral-limit-m 12 \
    --follow-yaw-limit-deg 120 \
    --image-width 768 \
    --image-height 768 \
    --rgb-jpeg-quality 95
```

To generate a detector-versus-reference report for an existing dataset:

```bash
python -m carla_data_collection.run_collection report-metrics \
    --output-dir /my_workspace/Resume/RAVP_Dataset
```

## Project Structure

```text
RAVP/
├── assets/
│   └── RAVP_demo.mp4
├── carla_data_collection/
│   ├── basic_dataset_eda.ipynb
│   ├── carla_utils.py
│   ├── collector.py
│   ├── config.py
│   ├── detector_3d.py
│   ├── ground_truth.py
│   ├── report_metrics.py
│   ├── run_collection.py
│   ├── utils.py
│   └── vision_detector.py
├── inference/
│   ├── carla_utils.py
│   ├── config.py
│   ├── mpc_controller.py
│   ├── pose_estimator.py
│   └── run_pursuit.py
├── pursuit_eval/
│   ├── batch_run.py
│   ├── config.py
│   ├── controller.py
│   ├── geometry.py
│   ├── metrics.py
│   ├── perception.py
│   ├── run.py
│   └── scenario.py
├── scripts/
│   ├── start_carla_watchdog.sh
│   ├── start_collect_train_towns.sh
│   └── train_target_pose_models.sh
├── target_pose_regression/
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── preprocessing.py
│   └── train.py
├── mmdet3d_models/
│   └── configs/
└── README.md
```

---

## Applications

| Area | Use |
|:-----|:----|
| Autonomous driving research | Study target-specific pursuit and identity-consistent following in traffic |
| Robotics and control | Combine learned perception with MPC for closed-loop target following |
| Vision-based tracking | Evaluate target initialization plus long-horizon mask-conditioned pursuit |
| Simulation-driven ML | Generate controlled per-target datasets for relative pose learning |

---

<div align="center">

Built around the goal of following the right car, not just the nearest one.

</div>
