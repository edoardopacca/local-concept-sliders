#!/usr/bin/env bash
# =============================================================================
# Push from the local machine to HPC (incremental: only new or modified files):
#   - every .slurm in <arch>/{tasks/<task>/,trained_sliders/training/}/jobs/new_slurm/
#   - every .yaml in <arch>/trained_sliders/training/configs/
#   - every .yaml in <arch>/trained_sliders/training/prompts/{old,new,test}_prompt/
#
# The repo structure is preserved: each file lands in the same relative
# folder on the HPC side.
#
# Usage:
#   ./tools/push_to_hpc.sh                # default: EVERYTHING (slurm + configs + prompts)
#   ./tools/push_to_hpc.sh slurm          # only .slurm (any jobs/new_slurm/ subfolder)
#   ./tools/push_to_hpc.sh configs        # only configs/*.yaml
#   ./tools/push_to_hpc.sh prompts        # only prompts/*/*.yaml
#   ./tools/push_to_hpc.sh yaml           # configs + prompts
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$SCRIPT_DIR/pull_config.sh" ]; then
    echo "ERROR: tools/pull_config.sh not found. Create it from pull_config.sh.example."
    exit 1
fi
source "$SCRIPT_DIR/pull_config.sh"

LOCAL_REPO_ABS="$(cd "$LOCAL_REPO" && pwd)"
cd "$LOCAL_REPO_ABS"

FILTER="${1:-all}"

# -----------------------------------------------------------------------------
# Build the list of files to push.
# -----------------------------------------------------------------------------
FILES=()

push_slurm() {
    while IFS= read -r f; do FILES+=( "$f" ); done < <(find . \
        -path '*/jobs/new_slurm/*.slurm' \
        -type f 2>/dev/null | sed 's|^\./||')
}

push_configs() {
    while IFS= read -r f; do FILES+=( "$f" ); done < <(find . \
        -path '*/trained_sliders/training/configs/*.yaml' \
        -type f 2>/dev/null | sed 's|^\./||')
}

push_prompts() {
    while IFS= read -r f; do FILES+=( "$f" ); done < <(find . \
        \( -path '*/trained_sliders/training/prompts/old_prompt/*.yaml' \
        -o -path '*/trained_sliders/training/prompts/new_prompt/*.yaml' \
        -o -path '*/trained_sliders/training/prompts/test_prompt/*.yaml' \) \
        -type f 2>/dev/null | sed 's|^\./||')
}

case "$FILTER" in
    all)
        push_slurm
        push_configs
        push_prompts
        ;;
    slurm)   push_slurm ;;
    configs) push_configs ;;
    prompts) push_prompts ;;
    yaml)
        push_configs
        push_prompts
        ;;
    *)
        echo "Invalid filter: $FILTER"
        echo "Use: all | slurm | configs | prompts | yaml"
        exit 1
        ;;
esac

echo "==============================================================="
echo "=== push_to_hpc: $FILTER -> HPC (incremental)               ==="
echo "==============================================================="

if [ ${#FILES[@]} -eq 0 ]; then
    echo "[warn] no files found for filter '$FILTER'"
    exit 0
fi

echo "[info] ${#FILES[@]} candidate files"
echo ""

# Unique directories (compatible with bash 3.2)
UNIQUE_DIRS=$(printf '%s\n' "${FILES[@]}" | while IFS= read -r f; do dirname "$f"; done | sort -u)

# Helper: count only files actually transferred (lines starting with
# `>f...` or `<f...`). Robust to empty strings (set -e safe).
_count_files() {
    local out="$1"
    if [ -z "$out" ]; then echo 0; return; fi
    printf '%s\n' "$out" | grep -cE '^[<>]f' 2>/dev/null || echo 0
}

TOTAL=0
while IFS= read -r dir; do
    [ -z "$dir" ] && continue
    src="$LOCAL_REPO_ABS/$dir/"
    dst="${HPC_USER}@${HPC_HOST}:${HPC_REPO}/$dir/"
    ssh "${HPC_USER}@${HPC_HOST}" "mkdir -p \"${HPC_REPO}/$dir\"" 2>/dev/null || true
    out=$(rsync -ahi --include='*.slurm' --include='*.yaml' --exclude='*' "$src" "$dst" 2>&1 | grep -E '^[<>]f' || true)
    n=$(_count_files "$out")
    if [ "$n" -gt 0 ]; then
        echo "[$(date +%H:%M:%S)] $dir/  ->  $n files:"
        echo "$out" | sed 's/^/    /'
        TOTAL=$((TOTAL + n))
    fi
done <<< "$UNIQUE_DIRS"

echo ""
if [ "$TOTAL" -eq 0 ]; then
    echo "[done] Everything already up to date, nothing to push."
else
    echo "[done] $TOTAL files pushed."
fi
