# Video Frame Interpolation (UpFrame) - Comprehensive Agent Context & Chat History

> **Document Purpose**: This file serves as an authoritative, complete context handover for any AI agent or human developer continuing work on this codebase. It documents the full project architecture, historical challenges, mathematical root causes of visual artifacts, critical bug fixes, operational constraints, and running instructions.

---

## 1. Executive Summary & Project Purpose

* **Repository**: `video_frame_interpolation` (`upframe`)
* **Remote Repository**: `git@github.com:tuananhlevan/video_frame_interpolation.git` (Branch: `main`)
* **Primary Target**: Ultra-high-quality Video Frame Interpolation (VFI) specialized for sports broadcasting footage (primarily football/soccer, 25 fps to 50 fps upframing at 1080p Full HD).
* **Core Philosophy**:
  1. Leverage state-of-the-art neural VFI backbones (RIFE v4.6, AMT-G, EMA-VFI with TTA).
  2. Maintain 100% trajectory and visual fidelity of fast-moving sports objects (footballs moving at 80–120 px/frame, sprinting cleats, swinging limbs).
  3. Suppress 1-frame temporal firefly spikes (boundary overshoots, optical flow singularities, attention activation spikes) without naive clamping that corrupts legitimate motion.
  4. Provide automated multi-GPU resource allocation and hardware-accelerated video encoding (NVENC / optimized libx264).

---

## 2. Codebase Architecture & Key Modules

```
video_frame_interpolation/
├── run_full_comparison_pipeline.sh   # Production script running RIFE -> AMT-G -> EMA-VFI + Deflicker
├── AGENT_CONTEXT.md                  # This handover document
├── setup.py / pyproject.toml         # UpFrame CLI package installation
├── configs/                          # YAML profile configurations (football_25_to_50.yaml, etc.)
├── weights/                          # Auto-downloaded model checkpoints (RIFE, AMT-G, EMA-VFI)
├── upframe/                          # Core engine package
│   ├── cli/
│   │   ├── args.py                   # Argument parser for CLI
│   │   └── upframe.py                # Main CLI entry point (`upframe <in> <out> [options]`)
│   ├── core/
│   │   ├── config.py                 # Dataclass configuration definitions
│   │   └── types.py                  # Core type definitions
│   ├── models/
│   │   ├── rife.py                   # RIFE v4.6 adapter
│   │   ├── amt.py                    # AMT-S / AMT-G adapter (All-Pairs Multi-Scale Motion Attention)
│   │   ├── ema_vfi.py                # EMA-VFI adapter with TTA support
│   │   ├── gmfss.py                  # GMFSS adapter
│   │   └── weights.py                # Auto-downloader, sha256 verifier, local fallback cache
│   ├── pipeline/
│   │   ├── anti_flicker.py           # TemporalAntiFlicker: GPU & CPU envelope clamping + ball protection
│   │   ├── postprocess.py            # Standalone GPU video deflickering tool
│   │   ├── ball_refiner.py           # Legacy heuristic ball inpainting (disabled by default)
│   │   ├── cadence.py                # 2:2 pulldown / duplicate cadence detection & removal
│   │   ├── decode.py                 # Video decoding engine
│   │   ├── encode.py                 # NVENC / libx264 FFmpeg video encoding engine
│   │   ├── resource_allocator.py     # Dynamic VRAM profiler and worker pool auto-sizer
│   │   ├── scene_detect.py           # Scene change cut detector (prevents cross-scene blending)
│   │   └── scheduler.py              # Chunking and multi-worker execution pipeline
│   └── workers/
│       ├── gpu_worker.py             # Isolated GPU worker process handling chunk VFI
│       └── pool.py                   # Worker pool manager
├── eval/                             # 4-Layer evaluation framework
│   ├── layer1_technical/             # PSNR, SSIM, LPIPS, Edge PSNR
│   ├── layer2_temporal/              # Warping error, temporal consistency, flicker metrics
│   ├── layer3_football/              # Ball detection, circularity, trajectory jitter, ghosting
│   ├── layer4_scorecard/             # Radar charts and multi-model benchmark aggregations
│   └── cli.py                        # Benchmark CLI (`python -m eval.cli benchmark`)
└── tests/                            # Comprehensive PyTest test suite (86 tests)
```

---

## 3. History of Issues Investigated, Root Causes & Fixes

### 3.1. EMA-VFI `No module named 'config'` on Google Colab / Fresh Environment
* **Symptom**: Running EMA-VFI failed during initialization with `ModuleNotFoundError: No module named 'config'`.
* **Root Cause**: EMA-VFI source code uses `import config as ema_cfg` assuming `backbones/EMA_VFI` is the current working directory. In a packaged environment or sub-module, python's `sys.path` did not contain the EMA-VFI root directory.
* **Resolution**: In `upframe/models/ema_vfi.py`, dynamically injected the module path into `sys.path` within a context manager during model construction, resolving the import cleanly without polluting global state.

### 3.2. Hardware Encoding (NVENC) vs. Software (libx264)
* **Symptom**: Users on various Linux machines and Google Colab saw warnings that `h264_nvenc` was not supported by `/usr/bin/ffmpeg`.
* **Root Cause**: Many standard package manager ffmpeg builds do not compile NVIDIA NVENC SDK headers.
* **Resolution**: 
  - In `run_full_comparison_pipeline.sh`, added automatic detection: `ffmpeg -encoders 2>/dev/null | grep -q "h264_nvenc"`. If detected, `--nvenc` is passed; otherwise it gracefully falls back to optimized `libx264` (`--no-nvenc`) with CRF=18 and fast preset.
  - In `upframe/pipeline/encode.py`, added runtime probe verification so ffmpeg will never abort due to invalid encoder flags.

### 3.3. GPU Memory Sizing & Dynamic Worker Allocation
* **Symptom**: On a 40GB A100 or 96GB Blackwell machine, either too few workers were spawned (underutilizing VRAM) or too many workers caused Out-Of-Memory (OOM).
* **Root Cause**: Static worker numbers or naive formulas based purely on total VRAM ignored per-worker chunk buffer size, optical flow tensor allocations during forward pass, and frame resolution.
* **Resolution**: Implemented `ResourceAllocator` in `upframe/pipeline/resource_allocator.py`:
  - Profiles model-specific peak memory usage per 1080p frame (e.g. AMT-G ~5.5GB, EMA-VFI ~4.8GB, RIFE ~2.2GB).
  - Automatically targets 50%–70% of available free VRAM, bounding workers safely to prevent CUDA OOM while achieving maximum throughput.

---

### 3.4. Deep Dive: The Sports VFI Artifacts (Ghost Ball, Bright Spots, and Fireflies)

This was the most critical problem solved in the project. Below is the complete mathematical and visual diagnosis:

#### Symptom 1: Violent 25Hz Ghost Ball Flickering ("Lỗi nháy nháy ghost ball")
* **User Observation**: The moving football flickered violently on/off across consecutive frames, looking like a phantom strobe ball.
* **Exact Root Cause**:
  1. A fast-moving football shot across the pitch moves at **$40 - 100\text{ pixels}$ per source frame** (e.g. frames 1255–1264: measured displacements of $80 - 95\text{ px}$).
  2. In the interpolated frame $F_{\text{inter}}$ ($t=0.5$), the ball is located at the midpoint $X_{\text{mid}}$, which is $20 - 50\text{ px}$ away from the ball in $F_0$ and $20 - 50\text{ px}$ away from the ball in $F_1$.
  3. The anti-flicker postprocessor (`postprocess.py`) and `TemporalAntiFlicker.process_tensor` used a spatial search window radius of $R = 10\text{ px}$ (or $16\text{ px}$).
  4. Because $20 - 50\text{ px} > R$, when max-pooling around $X_{\text{mid}}$ in $F_0$ and $F_1$, there was **no ball in either source frame within radius $R$**—only dark green pitch grass (luminance $\approx 80$).
  5. The football in $F_{\text{inter}}$ had luminance $> 220$.
  6. The anti-flicker algorithm misclassified the real football as an "unphysical outlier spike" ($220 > 80 + 20$) and **replaced the real ball with green grass blend `0.5 * (F0 + F1)`!**
  7. On even frames (source camera frames), the ball was present. On odd frames (interpolated), the ball was erased into grass. This produced a 25Hz blinking ghost ball.

#### Symptom 2: Bright Glowing Spots on the Pitch ("Đốm sáng trên cỏ") & Smudges
* **User Observation**: Odd white blobs and cyan/green smudges appeared on the grass (e.g., around players' feet and near sideline banners).
* **Exact Root Cause**:
  1. In earlier versions, `--ball-refine` (`BallRefiner`) had been enabled by default in `run_full_comparison_pipeline.sh`.
  2. `BallRefiner` used heuristic Hough/contour circle detection with relaxed circularity thresholds.
  3. On football footage, white sprinting shoes, socks, shin guards, and sideline pitch markings were frequently misidentified as balls.
  4. `BallRefiner` extracted patches and pasted synthetic ball circles onto the grass at expected midpoints $\implies$ creating persistent glowing white blobs ("đốm sáng").
  5. It then executed `cv2.inpaint(..., flags=cv2.INPAINT_TELEA)` to erase ghost halos, which smeared blurry patches across the grass (such as the cyan goalkeeper kit smudge in frame 1881).
  6. In contrast, pure neural networks (RIFE, AMT-G, EMA-VFI) without `BallRefiner` produced **flawless, smudge-free optical flow** without any bright spots.

#### Symptom 3: Ripping/Slitting Moving Player Limbs
* **User Observation**: Moving white graphics, jerseys, or sprinting cleats had dark vertical slits sliced through them in interpolated frames (e.g. frame 713).
* **Exact Root Cause**: A kicking player's foot swung at $39\text{ px}$ per source frame ($20\text{ px}$ at midpoint). The default `radius=10` in `postprocess.py` was too small to encompass the $20\text{ px}$ motion, causing the temporal envelope to clamp the middle of the shoe to background grass.

---

### 3.5. The Comprehensive Engineering Solution Implemented

1. **High-Speed Ball Trajectory Protection (`detect_ball_mask`)**:
   - In [upframe/pipeline/anti_flicker.py](file:///home/levantuananh/VDT_VT/video_frame_interpolation/upframe/pipeline/anti_flicker.py):
     - Added `detect_ball_mask(frame_0, frame_1)`.
     - Detects ball candidates in $F_0$ and $F_1$, filtering out false positives:
       - Candidates with a stationary counterpart in the adjacent frame ($< 20\text{ px}$) are rejected (these are player shoes/socks or pitch markings).
       - Candidates with mismatched radii ($> 3.5\text{ px}$ difference) are rejected.
       - Flying footballs with valid sports displacement ($25.0 \le \Delta \le 160.0\text{ px}$) are matched.
     - Generates a spatial trajectory capsule mask along the vector from $P_0$ to $P_1$ with radius $r = \max(r_0, r_1) + 4$.
     - Exempts this entire region from temporal envelope clamping:
       ```python
       # Tensor GPU flow:
       if ball_mask is not None:
           clamped = torch.where(ball_mask, batch_inter, clamped)
       ```
   - **Verification Result**:
     - Real football trajectory test on frames 1255–1256: `Mean diff between raw RIFE and anti-flicker: 0.0` (zero pixels modified, ball 100% preserved).
     - Isolated impulse noise test: White firefly sparks (intensity 255) on grass are 100% eliminated back to clean background `[30, 120, 80]`.

2. **Limb Motion Envelope Enlargement**:
   - In [upframe/pipeline/postprocess.py](file:///home/levantuananh/VDT_VT/video_frame_interpolation/upframe/pipeline/postprocess.py#L177):
     - Increased default spatial search radius from `10` to `16`.
     - Covers limb and foot motions up to $32\text{ px}$ displacement without clipping.
     - Tested on frame 713: Clamped value with radius 16 equals raw value `[200, 193, 199]` (exact match, no slits).

3. **Disabled Heuristic `BallRefiner` by Default**:
   - In [run_full_comparison_pipeline.sh](file:///home/levantuananh/VDT_VT/video_frame_interpolation/run_full_comparison_pipeline.sh):
     - Set `BALL_OPT=""` by default.
     - Prevents OpenCV Telea inpainting and synthetic patch pasting from generating false bright spots or smears on the pitch.

4. **Removed Evaluation Pipeline from Comparison Script**:
   - Per user instruction, completely removed Stage 4 (the 4-layer evaluation benchmark and scorecard generation) from [run_full_comparison_pipeline.sh](file:///home/levantuananh/VDT_VT/video_frame_interpolation/run_full_comparison_pipeline.sh).
   - The script now runs solely the core interpolation and GPU deflickering for the 3 models without wasting time on benchmark metrics.

### 3.6. Close-Up Player Tearing / 25Hz Flickering Artifacts (02:11 Artifact)
* **Symptom**: In close-up tracking shots of running players (e.g. at 02:11 in `highlight_test_rife.mp4`), dark jagged holes and chromatic fringes were carved into the yellow player jersey numbers, shoulders, and hair on interpolated frames, producing violent 25Hz strobing.
* **Root Cause**:
  1. Close-up player motion reached **155 px/frame** displacement (78 px at midpoint $t=0.5$).
  2. The static envelope radius ($R=16\text{ px}$) in `TemporalAntiFlicker` could not reach the player numbers in $F_0$ or $F_1$ (78 px away), seeing only black jersey.
  3. `TemporalAntiFlicker` misclassified the bright yellow numbers as single-frame firefly outliers and replaced them with `t_blend` (black jersey).
* **Resolution**:
  - Implemented **Morphological Scale Separation** in `TemporalAntiFlicker.process_tensor` (GPU) and `process` (CPU):
    - Applies morphological opening (erosion with $11\times 11$ kernel, dilation with $17\times 17$ kernel) to the violation mask.
    - Isolated single-frame firefly noise ($\le 4\text{ px}$) collapses to 0 and is 100% suppressed (preserving the white socks firefly fix).
    - Large continuous semantic structures (numbers, limbs, bodies) retain their solid cores and are 100% preserved from clamping (hole pixels dropped from 6,191 to 0).

---

## 4. Current State & Verification

### 4.1. Unit Test Suite
* **Command**: `/home/levantuananh/anaconda3/envs/vfi/bin/pytest tests/`
* **Status**: **84 passed, 2 skipped** (the 2 skipped tests are mock FFmpeg probe tests that require mock video assets).
* **Key Unit Tests Passing**:
  - `test_anti_flicker_process_tensor_preserves_fast_flying_ball`: Verifies 80px moving ball is 100% preserved while unprotected clamps to grass.
  - `test_anti_flicker_suppresses_severe_outliers`: Verifies impulse noise is suppressed when protect_ball=False.
  - `test_anti_flicker_preserves_fast_moving_socks`: Verifies 20px moving socks are preserved with radius=16.
  - `test_deflicker_video_e2e`: Verifies end-to-end video deflickering on mp4 video.
  - `test_backbone_adapters`: Verifies RIFE, AMT-G, EMA-VFI, and GMFSS adapters load and execute.

### 4.2. Git Status & Remote Sync
* **Branch**: `main`
* **Tracking**: `origin/main` (GitHub: `tuananhlevan/video_frame_interpolation`)
* **Latest Commits**:
  - `484942c`: `chore(pipeline): remove evaluation benchmark stage from comparison script`
  - `a216ee2`: `fix(anti-flicker): eliminate ball ghost flickering and bright spots by protecting flying ball trajectory and disabling heuristic ball-refine`

---

## 5. Critical Operating Rules & Constraints for Future Agents

> [!IMPORTANT]
> Any future agent working on this codebase MUST follow these operational constraints:

1. **Git Push Authentication**:
   - Always run git push with the user's active SSH agent socket:
     ```bash
     SSH_AUTH_SOCK=/run/user/1000/keyring/ssh git push origin main
     ```
   - Standard `git push` without this environment variable will fail with Permission Denied.

2. **Python Environment**:
   - Use the dedicated Conda environment located at:
     `/home/levantuananh/anaconda3/envs/vfi/bin/python`

3. **Never Re-enable `BallRefiner` by Default**:
   - Modern neural architectures (RIFE, AMT, EMA-VFI) already handle optical flow smoothly.
   - Do NOT pass `--ball-refine` in default scripts or production pipelines. It creates false-positive artifacts on sports cleats and pitch markings.

4. **Preserve Fast Ball Trajectory Protection**:
   - Any modifications to `TemporalAntiFlicker` must ensure `detect_ball_mask` remains intact in both `process_tensor` (GPU) and `process` (CPU).
   - Fast balls must never be clamped by spatial pooling envelopes.

---

## 6. How to Run the Pipeline

### 6.1. Running the Full 3-Model Comparison Pipeline
To run RIFE, AMT-G, and EMA-VFI + TTA with GPU postprocess deflickering:
```bash
./run_full_comparison_pipeline.sh [path/to/video.mp4]
```
Outputs produced:
- `<video_stem>_rife_clean.mp4`
- `<video_stem>_amtg_clean.mp4`
- `<video_stem>_emavfi_clean.mp4`

### 6.2. Running UpFrame CLI Directly
```bash
# Basic RIFE interpolation (25fps -> 50fps)
upframe input.mp4 output_rife.mp4 --model rife --cadence-filter

# AMT-G interpolation
upframe input.mp4 output_amtg.mp4 --model amt-g --cadence-filter

# EMA-VFI with Test-Time Augmentation (TTA)
upframe input.mp4 output_emavfi.mp4 --model ema-vfi --tta --cadence-filter
```

### 6.3. Running Standalone Postprocess Deflicker
```bash
python -m upframe.pipeline.postprocess -i input_50fps.mp4 -o cleaned_50fps.mp4 --radius 16 --device cuda:0
```

### 6.4. Running Evaluation Benchmark (Standalone)
```bash
python -m eval.cli benchmark \
    --source original_25fps.mp4 \
    --models "RIFE=rife_clean.mp4,AMTG=amtg_clean.mp4,EMAVFI=emavfi_clean.mp4" \
    --eval-dir evaluation_log/benchmark_run \
    --workers 2
```
