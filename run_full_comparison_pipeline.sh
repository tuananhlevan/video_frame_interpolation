#!/bin/bash
set -e

PYTHON="/home/levantuananh/anaconda3/envs/vfi/bin/python"
WORKDIR="/home/levantuananh/VDT_VT/video_frame_interpolation"
INPUT="highlight_test.mp4"

cd "$WORKDIR"

echo "=========================================================="
echo "Starting Full Sequential VFI Pipeline with Anti-Flicker"
echo "Input: $INPUT (13,151 frames, ~8m46s)"
echo "Sequence: 1) RIFE  ->  2) AMT-G  ->  3) EMA-VFI + TTA"
echo "Start time: $(date)"
echo "=========================================================="

# 1. RIFE
echo ""
echo "[Step 1/3] Running RIFE with ball-refine, cadence-filter, anti-flicker..."
$PYTHON -m upframe.cli.upframe "$INPUT" "highlight_test_rife_antiflicker.mp4" \
    --model rife \
    --ball-refine \
    --cadence-filter \
    --anti-flicker \
    --resume

echo "[Step 1/3] RIFE completed successfully at $(date)!"

# 2. AMT-G
echo ""
echo "[Step 2/3] Running AMT-G with ball-refine, cadence-filter, anti-flicker..."
$PYTHON -m upframe.cli.upframe "$INPUT" "highlight_test_amtg_antiflicker.mp4" \
    --model amt-g \
    --ball-refine \
    --cadence-filter \
    --anti-flicker \
    --resume

echo "[Step 2/3] AMT-G completed successfully at $(date)!"

# 3. EMA-VFI + TTA
echo ""
echo "[Step 3/3] Running EMA-VFI with TTA, ball-refine, cadence-filter, anti-flicker..."
$PYTHON -m upframe.cli.upframe "$INPUT" "highlight_test_emavfi_tta_antiflicker.mp4" \
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
