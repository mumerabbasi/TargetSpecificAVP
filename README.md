# RAVP

RAVP is a CARLA-based pipeline for learning relative target-vehicle pose from ego-view images and using that estimate inside a pursuit controller.

The core idea is:

1. collect rich supervision offline in CARLA,
2. train a much cheaper online pose model,
3. feed that pose estimate to downstream control.

The collection stack has now been redesigned around the exact downstream contract:

- input to the future CNN: one ego RGB frame plus one target-car mask,
- output from the CNN: the pose of that masked car only.

So the dataset is now explicitly per-target, not per-frame.

## Current status

The data-collection side is the most up-to-date part of the repository.

- `carla_data_collection/` now builds a per-target mask-conditioned dataset.
- `pose_estimation/` and `inference/` still contain older bbox-oriented assumptions in some places.
- The new collector already emits the assets needed for a mask-conditioned training pipeline:
  - shared ego RGB frames,
  - one mask per visible target vehicle,
  - separate GT and predicted pose CSVs.

## End-to-end pipeline

### Stage A: raw traffic capture

Raw capture is done in continuous CARLA traffic rather than by spawning isolated target scenes.

For each accepted frame, the collector saves:

- ego RGB image,
- full LiDAR point cloud,
- full CARLA instance-segmentation image,
- ego state,
- metadata for all visible vehicle actors.

This stage is reusable. Once raw capture exists, you can rebuild masks, GT labels, predicted labels, and detector benchmarks without recollecting CARLA.

### Stage B: build the GT per-target dataset

The GT dataset builder:

1. loads raw RGB, instance segmentation, and actor metadata,
2. enumerates visible vehicle targets in that frame,
3. generates one SAM3 mask per target,
4. saves one mask file per target,
5. writes one row per target to `gt_poses.csv`.

If one frame contains 3 visible valid cars, that frame produces 3 samples:

- same `rgb_path`,
- 3 different `mask_path`s,
- 3 different poses.

This mirrors inference exactly: the image can contain many cars, but the pose model only sees one target mask at a time.

### Stage C: attach predicted pose labels

Predicted labels are attached afterward, not during raw capture.

The detector stage:

1. runs a LiDAR 3D detector on the full-frame point cloud,
2. matches detections to the already accepted GT target actors,
3. writes the matched subset to `pred_poses.csv`,
4. reuses the exact same RGB and mask assets as the GT dataset.

This means:

- `gt_poses.csv` is the full accepted target set,
- `pred_poses.csv` is a subset that also has matched detector poses.

### Stage D: benchmark detectors

The benchmark stage evaluates detector candidates at the per-target sample level rather than only at raw detector-box level.

That matters because the real downstream question is:

“Given a crowded ego frame and one target mask, can the detector provide a useful pose label for that target?”

## Repository layout

```text
RAVP/
├── carla_data_collection/
│   ├── __init__.py
│   ├── benchmark_detectors.py
│   ├── carla_utils.py
│   ├── config.py
│   ├── dataset_builder.py
│   ├── detector_3d.py
│   ├── ground_truth.py
│   ├── preprocess_data.py
│   ├── raw_capture.py
│   ├── run_collection.py
│   └── vision_detector.py
├── inference/
├── mmdet3d_models/
│   ├── centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20220810_025930-657f67e0.pth
│   └── configs/
├── pose_estimation/
└── utils/
```

## What each collection module does

- `config.py`: shared configuration for raw capture, SAM3 mask generation, detector attachment, and benchmarking.
- `run_collection.py`: CLI entry point with separate subcommands for capture, GT build, prediction attachment, and benchmarking.
- `raw_capture.py`: stage A raw traffic capture in CARLA.
- `dataset_builder.py`: stage B GT build plus stage C prediction attachment.
- `vision_detector.py`: SAM3 wrapper.
- `detector_3d.py`: MMDetection3D wrapper currently configured around CenterPoint.
- `ground_truth.py`: relative-pose math and detection-to-actor matching helpers.
- `carla_utils.py`: CARLA world setup, traffic spawning, sensor synchronization, visibility filtering, and metadata creation.
- `preprocess_data.py`: merge multiple already-built per-target datasets into one dataset root.
- `benchmark_detectors.py`: evaluate detector candidates on accepted GT target samples.

## Environment split

CARLA, SAM3, and MMDetection3D do not share a single easy dependency stack on this machine, so the practical setup is split by stage.

### `ravp-carla37`

Use this env for raw CARLA capture only.

Reason:

- local CARLA 0.9.15 provides Python 3.7 wheels here.

### `ravp`

Use this env for:

- SAM3-backed GT dataset build,
- future mask-conditioned training work,
- general project-side Python tooling.

### `ravp-det`

Use this env for:

- `attach-predictions`,
- `benchmark-detectors`,
- MMDetection3D-based detector experiments.

## Collection outputs

The built dataset layout is:

```text
carla_dataset/
├── raw_capture/
│   ├── rgb/
│   ├── lidar/
│   ├── instance/
│   └── metadata/
├── rgb/
├── masks/
├── gt_poses.csv
├── pred_poses.csv
└── benchmarks/
```

### Raw capture assets

`capture-raw` writes:

- `raw_capture/rgb/frame_XXXXXX.png`
- `raw_capture/lidar/frame_XXXXXX.npy`
- `raw_capture/instance/frame_XXXXXX.npy`
- `raw_capture/metadata/frame_XXXXXX.json`

Each metadata JSON stores:

- `frame_id`, `episode_id`, `town`, `tick`
- relative raw asset paths
- ego pose and velocity
- camera metadata
- a `visible_actors` list containing one record per visible target vehicle
- capture behavior is controlled by `--traffic-mode`:
  - `traffic_manager` is the default for natural CARLA traffic
  - `constant_velocity` is a fallback for maps where Traffic Manager is unstable

### Final shared assets

The built dataset reuses frame assets and expands them into per-target samples:

- `rgb/frame_XXXXXX.png`: shared once per frame
- `masks/frame_XXXXXX_actor_<id>.png`: one mask per accepted target car

### Per-target CSV schema

Both `gt_poses.csv` and `pred_poses.csv` use the same schema:

| Column | Meaning |
| --- | --- |
| `sample_id` | Per-target sample id, currently `frameid_actorid` |
| `frame_id` | Shared RGB frame id |
| `episode_id` | Raw capture episode id |
| `town` | CARLA town name |
| `tick` | CARLA simulator tick |
| `actor_id` | CARLA GT vehicle id used as the identity anchor |
| `rgb_path` | Shared RGB frame path |
| `mask_path` | Target mask path |
| `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2` | Target mask bbox |
| `mask_area_px` | Target mask area in pixels |
| `dx_m`, `dy_m`, `dz_m`, `yaw_deg` | Relative pose in ego LiDAR coordinates |
| `pose_score` | `1.0` for GT rows, detector confidence for predicted rows |

Important policy:

- there are no prediction-error columns anymore,
- a frame may appear in many rows,
- `gt_poses.csv` may have more rows than `pred_poses.csv`,
- `pred_poses.csv` only contains targets that were accepted in GT and also matched by the detector.

## CLI workflow

Run commands from the repository root.

### 1. Capture raw traffic frames

```bash
conda activate ravp-carla37
cd /path/to/RAVP
python -m carla_data_collection.run_collection capture-raw \
  --output-dir ./carla_dataset \
  --carla-port 2150 \
  --towns Town01 Town02 Town03 \
  --target-samples-per-town 3000 \
  --max-frames-per-town 12000 \
  --max-episodes-per-town 4 \
  --num-traffic-vehicles 80 \
  --fresh
```

If a town crashes when Traffic Manager is initialized, rerun that town with:

```bash
python -m carla_data_collection.run_collection capture-raw \
  --output-dir ./carla_dataset \
  --towns Town03_Opt \
  --traffic-mode constant_velocity \
  --num-traffic-vehicles 40
```

### 2. Build shared RGB/masks plus GT labels

```bash
conda activate ravp
cd /path/to/RAVP
python -m carla_data_collection.run_collection build-gt-dataset \
  --output-dir ./carla_dataset \
  --sam3-repo-path /path/to/sam3 \
  --sam3-device cuda:0
```

### 3. Attach predicted detector labels

```bash
conda activate ravp-det
cd /path/to/RAVP
python -m carla_data_collection.run_collection attach-predictions \
  --output-dir ./carla_dataset \
  --detector-name centerpoint \
  --detector-device cuda:0
```

### 4. Benchmark detector candidates

```bash
conda activate ravp-det
cd /path/to/RAVP
python -m carla_data_collection.run_collection benchmark-detectors \
  --output-dir ./carla_dataset \
  --candidate voxelnext=/path/to/config.py::/path/to/checkpoint.pth
```

## Merging multiple built datasets

`carla_data_collection.preprocess_data` now merges multiple already-built per-target datasets.

Example:

```bash
python -m carla_data_collection.preprocess_data \
  --source /path/to/built_datasets \
  --dest /path/to/carla_dataset_merged
```

The merged dataset preserves:

- shared `rgb/` assets,
- shared `masks/` assets,
- `gt_poses.csv`,
- `pred_poses.csv`.

## Defaults and model assets

The repository now contains:

- vendored CenterPoint config files under `mmdet3d_models/configs/`,
- the default CenterPoint checkpoint under `mmdet3d_models/`.

Collection defaults live in `carla_data_collection/config.py`.

Important configurable areas include:

- CARLA host, port, and town list
- traffic density and episode limits
- image size and FOV
- capture-time visibility thresholds
- SAM3 paths and thresholds
- detector config/checkpoint/device
- distance, lateral, and yaw coverage bins

## Training and inference notes

The collection pipeline is already aligned with the intended mask-conditioned formulation.

The training and inference folders are still partly legacy:

- `pose_estimation/` still assumes bbox-oriented inputs in several places,
- `inference/` still mixes learned pose with CARLA-only signals for identity tracking and target speed.

So the new dataset is ahead of the current training stack. That is intentional: the collector is now producing the right data for the next training migration.

## Practical limitations

Important caveats:

1. CARLA raw capture depends on the local CARLA wheel ABI and is easiest to keep isolated.
2. SAM3 prompting is still an active part of the project and may need further tuning depending on scene scale and target visibility.
3. Predicted labels are only as good as the detector match quality on the accepted GT target set.
4. `pred_poses.csv` is expected to be smaller than `gt_poses.csv`.
5. Some CARLA maps can be unstable with Traffic Manager; `constant_velocity` capture is the fallback when that happens.
6. The online pursuit path is not yet fully perception-only because it still relies on simulator cues for some tracking logic.

## Recommended workflow

For current work, the cleanest path is:

1. collect raw traffic with `capture-raw`
2. build GT RGB/mask samples with `build-gt-dataset`
3. attach detector labels with `attach-predictions`
4. benchmark detectors if needed
5. train or migrate the mask-conditioned pose model
6. plug the trained model into pursuit
