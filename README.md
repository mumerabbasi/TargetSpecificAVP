# RAVP

RAVP is a CARLA-based pipeline for learning the relative pose of one target car from:

- one ego RGB image
- one mask for the target car

The downstream contract is explicit:

- input to the pose model: `rgb + target_mask`
- output from the pose model: `dx_m, dy_m, dz_m, yaw_follow_deg`

The repository uses one compact dataset pipeline and one canonical Python env.

## Current design

The primary workflow is:

1. start CARLA under a watchdog
2. run `collect-dataset` in `ravp`
3. for each accepted frame, generate final RGB, SAM3 masks, GT poses, and detector poses immediately
4. save only the final dataset assets

The default workflow writes the final training dataset directly.

## Canonical env

Use `ravp` for:

- CARLA client code
- SAM3
- MMDetection3D / detector inference
- dataset collection
- pursuit evaluation

`ravp` is the project env.

## Start CARLA

From inside the `CARLA/` directory, start CARLA with:

```bash
mkdir -p /tmp/runtime-1001 && chown 1001:1001 /tmp/runtime-1001 && \
HOME=/tmp XDG_RUNTIME_DIR=/tmp/runtime-1001 SDL_AUDIODRIVER=dummy DISPLAY= \
setpriv --reuid=1001 --regid=1001 --clear-groups \
./CarlaUE4.sh -opengl -RenderOffScreen -quality-level=Epic -carla-port=2150 -carla-streaming-port=2151 -nosound
```

The watchdog script uses this exact launch command and does not modify CARLA itself.

## Dataset layout

The compact dataset format is:

```text
dataset/
├── rgb/
├── masks/
├── gt_poses.csv
├── pred_poses.csv
├── frames.jsonl
└── collection_summary.json
```

Storage policy:

- RGB is stored once per accepted frame as `768x768` JPEG
- one binary mask PNG is stored per accepted target car
- `gt_poses.csv` and `pred_poses.csv` reuse the same RGB and mask assets
- full instance frames are not saved
- raw LiDAR caches are not saved

## Sample semantics

The dataset is per-target, not per-frame.

If one ego frame contains 3 accepted cars, that frame produces 3 datapoints:

- same `rgb_path`
- 3 different `mask_path`s
- 3 different pose labels

This mirrors inference exactly: the model only predicts the pose of the car whose mask is provided.

## CSV schema

Both `gt_poses.csv` and `pred_poses.csv` use the same columns:

- `sample_id`
- `frame_id`
- `episode_id`
- `town`
- `tick`
- `actor_id`
- `rgb_path`
- `mask_path`
- `bbox_x1`
- `bbox_y1`
- `bbox_x2`
- `bbox_y2`
- `mask_area_px`
- `dx_m`
- `dy_m`
- `dz_m`
- `yaw_deg`
- `yaw_follow_deg`
- `follow_valid`
- `pose_score`

Meaning:

- `yaw_deg` is the raw relative yaw
- `yaw_follow_deg` is the follow-regime folded yaw in `[-90, 90]`
- `follow_valid` marks samples that fit the pursuit-style forward-follow assumption
- `pose_score` is `1.0` for GT rows and detector confidence for predicted rows

## Collection command

Single-pass collection:

```bash
conda activate ravp
python -m carla_data_collection.run_collection collect-dataset \
  --output-dir /my_workspace/Resume/RAVP_Dataset_Compact \
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

`Town10HD` and `Town10HD_Opt` are the default holdout family for generalization.

## Resume and retry

The collector resumes from `frames.jsonl` and reconciles:

- `gt_poses.csv`
- `pred_poses.csv`
- `rgb/`
- `masks/`

so interrupted runs can continue cleanly.

Retry scripts:

- `scripts/start_carla_watchdog.sh`
- `scripts/start_collect_train_towns.sh`

The collection launcher retries after collector failure or CARLA restart and skips the held-out `Town10HD` family.

## Reporting

Write a detailed detector-vs-GT report for an existing compact dataset:

```bash
conda activate ravp
python -m carla_data_collection.run_collection report-metrics \
  --output-dir /my_workspace/Resume/RAVP_Dataset_Compact
```

This writes `detailed_metrics.json` in the dataset root.

## Perception in collection

Collection behavior per accepted frame:

1. capture synchronized RGB, instance segmentation, and LiDAR in memory
2. enumerate visible target cars
3. run SAM3 immediately on the RGB frame with per-actor box prompts
4. use in-memory CARLA instance masks only to assign SAM3 masks to GT actor identities
5. run the 3D detector once on the full LiDAR frame
6. match detections back to the accepted GT actors
7. write GT and predicted rows immediately

Nothing is deferred to a later raw replay stage.

## Pursuit evaluation

`pursuit_eval/` runs SAM3 and the detector in-process in `ravp`.

The pursuit loop:

- instantiates SAM3 and the detector once
- tracks the target online with SAM3
- reseeds with a bbox when needed
- estimates target pose directly from the current LiDAR frame
- feeds that pose into the pursuit controller

## Key modules

```text
carla_data_collection/
├── carla_utils.py
├── collector.py
├── config.py
├── detector_3d.py
├── ground_truth.py
├── preprocess_data.py
├── report_metrics.py
├── run_collection.py
├── utils.py
└── vision_detector.py

pursuit_eval/
├── batch_run.py
├── config.py
├── controller.py
├── geometry.py
├── metrics.py
├── perception.py
├── run.py
└── scenario.py
```

Module roles:

- `collector.py`: the single-pass compact dataset collector
- `vision_detector.py`: SAM3 image wrapper used during collection
- `detector_3d.py`: MMDetection3D wrapper used during collection
- `report_metrics.py`: detector-vs-GT dataset reporting
- `perception.py`: in-process SAM3 tracker and detector pose source for pursuit eval
- `run.py`: closed-loop pursuit evaluation entry point

## Practical note on size

The compact layout was introduced specifically to keep training data size under control.

The main size wins are:

- `768x768` JPEG instead of `1024x1024` PNG for RGB
- no saved raw instance images
- no saved raw LiDAR caches
- no duplicate intermediate assets

This keeps the saved dataset aligned with the actual downstream training contract instead of archiving full raw CARLA sensor dumps.
