# Upframe: Production-Grade Video Upframing Pipeline

A production-grade, model-agnostic, multi-GPU video frame insertion (VFI) pipeline designed for sports and broadcast video (primarily 25 fps football/soccer footage) to produce rock-solid 50 fps output with AI-generated intermediate frames.

---

## Key Features

- **Simple CLI**: `upframe input.mp4 output_50fps.mp4`
- **Multi-GPU Parallel Worker Pool**: Splits videos into overlapping chunks and processes them across independent GPU workers (e.g., 3× NVIDIA L40S 48GB). Gracefully falls back to single GPU or CPU.
- **Model-Agnostic Architecture**: Pluggable `BaseVFIModel` interface with integrated support for models organized under `backbones/`:
  - **RIFE** (`backbones/rife/`): Practical-RIFE / IFNet - default production baseline
  - **AMT** (`backbones/amt/`): All-Pairs Multi-Scale Motion Estimation
  - **IFRNet** (`backbones/ifrnet/`): Intermediate Feature Refine Network
  - **FILM** (`backbones/film/`): Frame Interpolation for Large Motion
  - **Blend**: Zero-dependency linear blend for testing and latency benchmarking
- **Scene-Cut Protection**: Content-aware shot boundary detection prevents hybrid blur artifacts across camera transitions.
- **Audio & Timing Integrity**:
  - Stream-copies original AAC audio (`-c:a copy`) without lossy re-encoding.
  - Normalizes non-standard container timing (~25.02 fps) into an exact 50.000 fps broadcast timeline without drift.
- **Fault-Tolerant Scheduling & Crash Recovery**:
  - Automatically tracks chunk progress in `pipeline_state.json`.
  - Supports `--resume` to retry failed chunks or resume interrupted jobs without reprocessing completed chunks.
- **Quality Control & Reporting**:
  - Generates the formal **UPFRAME REPORT** with real-time throughput factor, frame counts, scene cuts, and codec details.

---

## Installation

### In Conda Environment (`vfi`)

```bash
conda activate vfi
pip install -e .
```

### Docker

```bash
docker build -t upframe:latest -f docker/Dockerfile .
```

---

## Usage

### Simplest Interface
```bash
upframe input.mp4 output_50fps.mp4
```

### Advanced Multi-GPU Production Run
```bash
upframe \
  --input match.mp4 \
  --output match_50fps.mp4 \
  --model rife \
  --gpus 0,1,2 \
  --chunk-size 1000 \
  --scene-threshold 0.35 \
  --nvenc \
  --crf 18
```

### Resuming an Interrupted Job
```bash
upframe -i match.mp4 -o match_50fps.mp4 --resume
```

### Fast Test / Benchmark Mode (using Blend model)
```bash
upframe output_test.mp4 /tmp/output_50fps.mp4 --model blend
```

---

## Quality Control Report Example

```
==================================================
               UPFRAME REPORT
==================================================
Input:             output_test.mp4
Output:            output_50fps.mp4

Resolution:        1920x1080
Input FPS:         ~25.00 fps
Output FPS:        50.00 fps
Duration:          00:25

Model:             RIFE
GPUs:              0,1,2

Source frames:     648
Generated frames:  647
Total out frames:  1,295
Scene cuts:        2

Processing time:   00:23 (23.4s)
Realtime factor:   0.94x

Audio:             stream copied
Encoding:          H.264
Status:            SUCCESS
==================================================
```

---

## Evaluation Pipeline (`eval`)

The repository includes a production-grade, 4-layer evaluation framework specifically designed for football broadcast VFI (25 FPS $\to$ 50 FPS). It operates on a **single-pass streaming sliding-window architecture** ($< 150\text{ MB}$ RAM footprint), preventing memory exhaustion even on full 90-minute matches.

All evaluation artifacts and logs are automatically organized inside the **`evaluation_log/`** directory.

### 1. Evaluate a Video (`run`)

Evaluate an interpolated 50 FPS video against its 25 FPS source:

```bash
# Basic run: automatically outputs to evaluation_log/<output_video_name>/
python -m eval.cli run \
  --source test_videos/output_test.mp4 \
  --output test_videos/output_test_interpolated_amt-l.mp4 \
  --model amt-l
```

#### Common Options:
* `--max-frames <N>`: Limit the number of source frames to evaluate for quick testing (e.g. `--max-frames 50`).
* `--sample-stride <N>`: Evaluate every $N$-th frame interval (default: `1`).
* `--processing-time <sec>`: Production processing time in seconds measured on GPU cluster (calculates Realtime Factor $\text{RTF} = \frac{\text{Processing Time}}{\text{Duration}}$ against the $1.0\times - 1.3\times$ broadcast target).
* `--human-mos-file <path.json>`: Path to JSON file containing human MOS panel scores (otherwise computes automated visual quality proxy).
* `--eval-dir <dir>`: Custom output folder (defaults to `evaluation_log/<output_video_name>/`).
* `--no-visuals`: Disable generating slow-motion clips, difference heatmaps, and frame extractions.

### 2. Side-by-Side Model Benchmark (`benchmark`)

Compare multiple candidate models side-by-side on the same source video:

```bash
python -m eval.cli benchmark \
  --source test_videos/output_test.mp4 \
  --models rife=test_videos/output_test_interpolated_rife.mp4,amtl=test_videos/output_test_interpolated_amt-l.mp4 \
  --max-frames 50
```
Outputs a comparative terminal matrix and stores sub-reports in `evaluation_log/benchmark_<source_name>/`.

### 3. Scientific Ground-Truth Benchmark (`ground-truth`)

When high-FPS reference footage (e.g., native 50 FPS / 100 FPS) is available, compute full-reference PSNR, SSIM, MS-SSIM, and LPIPS:

```bash
python -m eval.cli ground-truth \
  --gt path/to/50fps_ground_truth.mp4 \
  --output path/to/model_50fps.mp4 \
  --max-frames 100
```

### 4. Inspect Visual Materials (`inspect`)

Generate slow-motion review clips (2x, 4x, 8x), extracted intermediate frames, and amplified difference heatmaps:

```bash
python -m eval.cli inspect \
  --source test_videos/output_test.mp4 \
  --output test_videos/output_test_interpolated_amt-l.mp4 \
  --slowmo 2,4
```

### 5. Probe Video Properties (`probe`)

Inspect stream properties, container timing, nominal/average FPS, and audio streams:

```bash
python -m eval.cli probe test_videos/output_test.mp4
```

---

### Output Artifacts Structure

Each evaluation generates a structured directory inside `evaluation_log/`:

```text
evaluation_log/<video_name>/
├── report.html                  # Interactive HTML dashboard (gauges, radar chart, breakdown)
├── report.json                  # Machine-readable evaluation results dataclass
├── metrics.csv                  # Tabular metrics and frame-by-frame stats
├── summary.txt                  # Terminal-formatted scorecard summary
├── evaluation.log               # Full timestamped execution log
├── suspicious/
│   └── suspicious_timestamps.csv # Anomaly moments flagged with broadcast timecodes (HH:MM:SS.mmm)
├── injected_frames/             # Extracted intermediate odd frames
├── diff_maps/                   # Colorized JET difference heatmaps
└── slowmo/                      # 2x and 4x slow-motion review clips
```

---

### Programmatic Python API

You can also run evaluations directly from Python:

```python
from eval import EvaluationPipeline, EvaluationConfig

config = EvaluationConfig(
    max_frames=100,
    generate_visuals=True
)
evaluator = EvaluationPipeline(config=config)

report = evaluator.evaluate(
    source_path="match_25fps.mp4",
    output_path="match_50fps.mp4",
    model_name="amt-l"
)

print(f"Quality Score: {report.scorecard.quality_score:.1f} / 10.0 ({report.scorecard.quality_status})")
print(f"Recommendation: {report.scorecard.recommendation}")
```

---

## Running Tests

Run the full unit test suite (including pipeline, layers, and streaming checks):

```bash
pytest -v tests/
```

