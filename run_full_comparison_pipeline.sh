#!/usr/bin/env bash
set -e

# ==============================================================================
# UpFrame Full Sequential Comparison Pipeline
#
# Runs 3 top VFI backbones (RIFE, AMT-G, EMA-VFI + TTA), applies GPU temporal
# postprocess deflickering to eliminate firefly spikes, and computes a 4-layer
# side-by-side benchmark comparison scorecard.
#
# Usage:
#   ./run_full_comparison_pipeline.sh [path/to/video.mp4] [options]
# Options:
#   --no-ball-refine   Disable football trajectory ball refiner (enabled by default)
#   --skip-eval        Skip final 4-layer evaluation benchmark
#   --device <dev>     Force compute device (default: cuda:0 if available, else cpu)
# ==============================================================================

# Prevent CUDA memory allocator fragmentation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Resolve Python executable
if [ -n "$VIRTUAL_ENV" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON_CMD="$VIRTUAL_ENV/bin/python"
elif [ -n "$CONDA_PREFIX" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    PYTHON_CMD="$CONDA_PREFIX/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

# Resolve UpFrame CLI
if command -v upframe &>/dev/null; then
    UPFRAME="upframe"
else
    UPFRAME="$PYTHON_CMD -m upframe.cli.upframe"
fi

POSTPROCESS="$PYTHON_CMD -m upframe.pipeline.postprocess"

# Parse arguments
INPUT=""
BALL_OPT="--ball-refine"
RUN_EVAL=true
DEVICE="cuda:0"

# Check GPU availability
if ! $PYTHON_CMD -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE="cpu"
fi

while [ $# -gt 0 ]; do
    case "$1" in
        --no-ball-refine)
            BALL_OPT=""
            shift
            ;;
        --ball-refine)
            BALL_OPT="--ball-refine"
            shift
            ;;
        --skip-eval|--no-eval)
            RUN_EVAL=false
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        -*)
            echo "Unknown option: $1"
            shift
            ;;
        *)
            if [ -z "$INPUT" ]; then
                INPUT="$1"
            fi
            shift
            ;;
    esac
done

# Default input video if not provided
if [ -z "$INPUT" ]; then
    if [ -f "highlight_test.mp4" ]; then
        INPUT="highlight_test.mp4"
    elif [ -f "test_videos/highlight_test.mp4" ]; then
        INPUT="test_videos/highlight_test.mp4"
    else
        INPUT="highlight_test.mp4"
    fi
fi

if [ ! -f "$INPUT" ]; then
    echo "Error: Input video not found at: $INPUT"
    echo "Usage: $0 [path/to/video.mp4] [--ball-refine] [--skip-eval]"
    exit 1
fi

INPUT_STEM="$(basename "$INPUT" | sed 's/\.[^.]*$//')"

# Check NVENC support in ffmpeg
if ffmpeg -encoders 2>/dev/null | grep -q "h264_nvenc"; then
    NVENC_OPT="--nvenc"
    NVENC_POST=""
    echo "Hardware Encoder: NVIDIA NVENC detected and enabled."
else
    NVENC_OPT=""
    NVENC_POST="--no-nvenc"
    echo "Hardware Encoder: 'h264_nvenc' not compiled into ffmpeg. Using optimized libx264."
fi

echo "=========================================================="
echo " Starting Full Comparison Pipeline with Postprocess Deflicker"
echo " CLI Command  : $UPFRAME"
echo " Python Bin   : $PYTHON_CMD"
echo " Input Video  : $INPUT"
echo " Target Device: $DEVICE"
echo " Ball Refine  : $([ -n "$BALL_OPT" ] && echo 'Enabled' || echo 'Disabled (safe mode)')"
echo " Benchmark QC : $([ "$RUN_EVAL" = true ] && echo 'Enabled' || echo 'Skipped')"
echo " Models       : 1) RIFE  ->  2) AMT-G  ->  3) EMA-VFI + TTA"
echo " Start time   : $(date)"
echo "=========================================================="

# ------------------------------------------------------------------------------
# 1. RIFE
# ------------------------------------------------------------------------------
echo ""
echo "=========================================================="
echo "[Stage 1/3] RIFE: Fast optical flow baseline"
echo "=========================================================="
RAW_RIFE="${INPUT_STEM}_rife_raw.mp4"
CLEAN_RIFE="${INPUT_STEM}_rife_clean.mp4"

$UPFRAME "$INPUT" "$RAW_RIFE" \
    --model rife \
    $NVENC_OPT \
    $BALL_OPT \
    --cadence-filter \
    --anti-flicker \
    --resume

echo "[Stage 1/3] Running GPU Postprocess Deflicker on RIFE output..."
$POSTPROCESS \
    -i "$RAW_RIFE" \
    -o "$CLEAN_RIFE" \
    --device "$DEVICE" \
    $NVENC_POST

echo "[Stage 1/3] RIFE finished -> Clean output: $CLEAN_RIFE ($(date))"

# ------------------------------------------------------------------------------
# 2. AMT-G
# ------------------------------------------------------------------------------
echo ""
echo "=========================================================="
echo "[Stage 2/3] AMT-G: All-Pairs Multi-Scale Motion Attention"
echo "=========================================================="
RAW_AMTG="${INPUT_STEM}_amtg_raw.mp4"
CLEAN_AMTG="${INPUT_STEM}_amtg_clean.mp4"

$UPFRAME "$INPUT" "$RAW_AMTG" \
    --model amt-g \
    $NVENC_OPT \
    $BALL_OPT \
    --cadence-filter \
    --anti-flicker \
    --resume

echo "[Stage 2/3] Running GPU Postprocess Deflicker on AMT-G output..."
$POSTPROCESS \
    -i "$RAW_AMTG" \
    -o "$CLEAN_AMTG" \
    --device "$DEVICE" \
    $NVENC_POST

echo "[Stage 2/3] AMT-G finished -> Clean output: $CLEAN_AMTG ($(date))"

# ------------------------------------------------------------------------------
# 3. EMA-VFI + TTA
# ------------------------------------------------------------------------------
echo ""
echo "=========================================================="
echo "[Stage 3/3] EMA-VFI + TTA: Inter-Frame Cross Attention"
echo "=========================================================="
RAW_EMAVFI="${INPUT_STEM}_emavfi_tta_raw.mp4"
CLEAN_EMAVFI="${INPUT_STEM}_emavfi_clean.mp4"

$UPFRAME "$INPUT" "$RAW_EMAVFI" \
    --model ema-vfi \
    $NVENC_OPT \
    --tta \
    $BALL_OPT \
    --cadence-filter \
    --anti-flicker \
    --resume

echo "[Stage 3/3] Running GPU Postprocess Deflicker on EMA-VFI output..."
$POSTPROCESS \
    -i "$RAW_EMAVFI" \
    -o "$CLEAN_EMAVFI" \
    --device "$DEVICE" \
    $NVENC_POST

echo "[Stage 3/3] EMA-VFI finished -> Clean output: $CLEAN_EMAVFI ($(date))"

# ------------------------------------------------------------------------------
# 4. 4-Layer Comparative Evaluation & Scorecard
# ------------------------------------------------------------------------------
if [ "$RUN_EVAL" = true ]; then
    echo ""
    echo "=========================================================="
    echo "[Stage 4/4] Executing 4-Layer Comparison Benchmark"
    echo "Comparing: RIFE vs. AMT-G vs. EMA-VFI"
    echo "=========================================================="
    
    BENCHMARK_DIR="evaluation_log/benchmark_${INPUT_STEM}"
    $PYTHON_CMD -m eval.cli benchmark \
        --source "$INPUT" \
        --models "RIFE=${CLEAN_RIFE},AMT-G=${CLEAN_AMTG},EMA-VFI=${CLEAN_EMAVFI}" \
        --eval-dir "$BENCHMARK_DIR" \
        --workers 2

    echo ""
    echo "Evaluation reports saved to: $BENCHMARK_DIR"
fi

echo ""
echo "=========================================================="
echo " FULL PIPELINE COMPLETED SUCCESSFULLY AT $(date)!"
echo " Outputs:"
echo "   1) RIFE Clean   : $CLEAN_RIFE"
echo "   2) AMT-G Clean  : $CLEAN_AMTG"
echo "   3) EMA-VFI Clean: $CLEAN_EMAVFI"
if [ "$RUN_EVAL" = true ]; then
    echo " Benchmark Scorecard: evaluation_log/benchmark_${INPUT_STEM}"
fi
echo "=========================================================="
