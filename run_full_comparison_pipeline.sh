#!/usr/bin/env bash
set -e

# Use installed 'upframe' CLI command directly (configured via pyproject.toml: [project.scripts])
# Fall back to python module execution if 'upframe' is not yet in PATH
if command -v upframe &>/dev/null; then
    UPFRAME="upframe"
elif command -v python3 &>/dev/null; then
    UPFRAME="python3 -m upframe.cli.upframe"
else
    UPFRAME="python -m upframe.cli.upframe"
fi

# Input video argument (defaults to highlight_test.mp4)
INPUT="${1:-highlight_test.mp4}"

if [ ! -f "$INPUT" ]; then
    echo "Error: Input video not found at: $INPUT"
    echo "Usage: $0 [path/to/video.mp4]"
    exit 1
fi

INPUT_STEM="$(basename "$INPUT" | sed 's/\.[^.]*$//')"

echo "=========================================================="
echo "Starting Full Sequential VFI Pipeline with Anti-Flicker"
echo "CLI Command: $UPFRAME"
echo "Input Video: $INPUT"
echo "Sequence: 1) RIFE  ->  2) AMT-G  ->  3) EMA-VFI + TTA"
echo "Start time: $(date)"
echo "=========================================================="

# 1. RIFE
echo ""
echo "[Step 1/3] Running RIFE with ball-refine, cadence-filter, anti-flicker..."
$UPFRAME "$INPUT" "${INPUT_STEM}_rife_antiflicker.mp4" \
    --model rife \
    --ball-refine \
    --cadence-filter \
    --anti-flicker \
    --resume

echo "[Step 1/3] RIFE completed successfully at $(date)!"

# 2. AMT-G
echo ""
echo "[Step 2/3] Running AMT-G with ball-refine, cadence-filter, anti-flicker..."
$UPFRAME "$INPUT" "${INPUT_STEM}_amtg_antiflicker.mp4" \
    --model amt-g \
    --ball-refine \
    --cadence-filter \
    --anti-flicker \
    --resume

echo "[Step 2/3] AMT-G completed successfully at $(date)!"

# 3. EMA-VFI + TTA
echo ""
echo "[Step 3/3] Running EMA-VFI with TTA, ball-refine, cadence-filter, anti-flicker..."
$UPFRAME "$INPUT" "${INPUT_STEM}_emavfi_tta_antiflicker.mp4" \
    --model ema-vfi \
    --tta \
    --ball-refine \
    --cadence-filter \
    --anti-flicker \
    --resume

echo "[Step 3/3] EMA-VFI + TTA completed successfully at $(date)!"

echo "=========================================================="
echo "ALL 3 MODELS COMPLETED SUCCESSFULLY AT $(date)!"
echo "=========================================================="
