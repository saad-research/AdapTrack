#!/usr/bin/env bash
# train_pool.sh -- one Colab session's worth of pool training.
#
#   bash scripts/train_pool.sh                    # TinyLlama pool
#   BASE=Qwen/Qwen2.5-1.5B TAG=qwen bash scripts/train_pool.sh
#
# Safe to re-run after a disconnect: train_pool.py skips finished adapters.
set -euo pipefail

DRIVE="${DRIVE:-/content/drive/MyDrive/adaptrack}"
BASE="${BASE:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
TAG="${TAG:-tinyllama}"
N_EXAMPLES="${N_EXAMPLES:-2500}"
PER_FAMILY="${PER_FAMILY:-24}"
LIMIT="${LIMIT:-}"

# Colab preinstalls torchao 0.10.0, which breaks peft 0.20.0 with an
# ImportError. This must run in EVERY session -- put it here, not in a setup
# script you might skip on a reconnect.
pip uninstall -y torchao >/dev/null 2>&1 || true

python - <<'PY'
import torch
assert torch.cuda.is_available(), "no GPU in this session"
name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
print(f"GPU: {name} (sm_{major}{minor})")
print("dtype will be", "bf16" if major >= 8 else "fp16 (no bf16 tensor cores)")
PY

if [ ! -d "$(dirname "$DRIVE")" ]; then
  echo "ERROR: Drive not mounted. Run in a cell first:"
  echo "  from google.colab import drive; drive.mount('/content/drive')"
  exit 1
fi

OUT="$DRIVE/pool_$TAG"
mkdir -p "$OUT"

python scripts/train_pool.py \
  --base "$BASE" \
  --out "$OUT" \
  --n-examples "$N_EXAMPLES" \
  --per-family "$PER_FAMILY" \
  ${LIMIT:+--limit "$LIMIT"} \
  2>&1 | tee -a "$OUT/train_log.txt"

echo
echo "pool at $OUT"
du -sh "$OUT" 2>/dev/null || true