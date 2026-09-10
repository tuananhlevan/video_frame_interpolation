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

## Running Tests

```bash
pytest -v tests/
```
