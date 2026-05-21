#!/usr/bin/env bash
# =============================================================================
# Pull from HPC to the local machine (incremental: only new / modified files):
#   - trained sliders   ->  <arch>/trained_sliders/sliders/
#   - image outputs     ->  <arch>/tasks/<task>/outputs/
#
# The repo structure is preserved: each file lands in the same relative
# folder it lives in on HPC.
#
# Usage:
#   ./tools/pull_from_hpc.sh                # download everything (default)
#   DRY_RUN=1 ./tools/pull_from_hpc.sh      # preview only
#   REMOVE_REMOTE=1 ./tools/pull_from_hpc.sh # download AND free HPC
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$SCRIPT_DIR/pull_config.sh" ]; then
    echo "ERROR: tools/pull_config.sh not found. Create it from pull_config.sh.example."
    exit 1
fi
source "$SCRIPT_DIR/pull_config.sh"

DRY_RUN_FLAG=""
[[ "${DRY_RUN:-0}" == "1" ]] && DRY_RUN_FLAG="--dry-run"

REMOVE_FLAG=""
[[ "${REMOVE_REMOTE:-0}" == "1" ]] && REMOVE_FLAG="--remove-source-files"

LOCAL_REPO_ABS="$(cd "$LOCAL_REPO" && pwd)"

echo "==============================================================="
echo "=== pull_from_hpc: sliders + outputs (incremental)          ==="
echo "==============================================================="

TOTAL=0

# Helper: count only files actually transferred (lines starting with
# `>f...` or `<f...`), excluding directory lines (`.d...` / `cd...`)
# that are just timestamp updates.
_count_files() {
    local out="$1"
    if [ -z "$out" ]; then echo 0; return; fi
    printf '%s\n' "$out" | grep -cE '^[<>]f' 2>/dev/null || echo 0
}

# --- 1. Trained sliders (sdxl + flux) ---
for arch in sdxl flux; do
    src="${HPC_USER}@${HPC_HOST}:${HPC_REPO}/${arch}/trained_sliders/sliders/"
    dst="${LOCAL_REPO_ABS}/${arch}/trained_sliders/sliders/"
    out=$(rsync -ahi $DRY_RUN_FLAG $REMOVE_FLAG \
        --exclude='__pycache__/' --exclude='_peft_cache/' \
        --include='*/' --include='*.safetensors' --include='*.pt' \
        --include='*.npy' --include='metadata.json' --exclude='*' \
        "$src" "$dst" 2>&1 | grep -E '^[<>ch.]' || true)
    n=$(_count_files "$out")
    if [ "$n" -gt 0 ]; then
        echo "[$(date +%H:%M:%S)] $arch sliders ($n files):"
        printf '%s\n' "$out" | grep -E '^[<>]f' | sed 's/^/    /'
        TOTAL=$((TOTAL + n))
    fi
done

# --- 2. Image outputs for every task (sdxl + flux) ---
for arch in sdxl flux; do
    src="${HPC_USER}@${HPC_HOST}:${HPC_REPO}/${arch}/tasks/"
    dst="${LOCAL_REPO_ABS}/${arch}/tasks/"
    out=$(rsync -ahi $DRY_RUN_FLAG $REMOVE_FLAG \
        --exclude='__pycache__/' --exclude='_peft_cache/' --exclude='logs/' \
        --include='*/' \
        --include='outputs/**/*.png' --include='outputs/**/*.jpg' \
        --include='outputs/**/*.json' --include='outputs/**/*.npy' \
        --exclude='*' \
        "$src" "$dst" 2>&1 | grep -E '^[<>ch.]' || true)
    n=$(_count_files "$out")
    if [ "$n" -gt 0 ]; then
        echo "[$(date +%H:%M:%S)] $arch outputs ($n files):"
        printf '%s\n' "$out" | grep -E '^[<>]f' | sed 's/^/    /'
        TOTAL=$((TOTAL + n))
    fi
done

echo ""
if [ "$TOTAL" -eq 0 ]; then
    echo "[done] Everything already up to date, nothing to download."
else
    echo "[done] $TOTAL files downloaded."
fi
[[ -n "$REMOVE_FLAG" ]] && echo "[note] source files removed from HPC."
