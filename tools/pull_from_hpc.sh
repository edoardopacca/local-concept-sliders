#!/usr/bin/env bash
# =============================================================================
# pull_from_hpc.sh
# =============================================================================
# Scarica da HPC al Mac (incrementale: solo file nuovi o modificati):
#   - sliders trainati  →  <arch>/trained_sliders/sliders/
#   - outputs immagini  →  <arch>/tasks/<task>/outputs/
#
# Mantiene la stessa struttura della repo: ogni file finisce nella stessa
# cartella relativa da cui parte su HPC.
#
# Uso:
#   ./tools/pull_from_hpc.sh                # scarica tutto (default)
#   DRY_RUN=1 ./tools/pull_from_hpc.sh      # anteprima
#   REMOVE_REMOTE=1 ./tools/pull_from_hpc.sh # scarica + libera HPC
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$SCRIPT_DIR/pull_config.sh" ]; then
    echo "ERROR: tools/pull_config.sh non trovato. Crealo da pull_config.sh.example."
    exit 1
fi
source "$SCRIPT_DIR/pull_config.sh"

DRY_RUN_FLAG=""
[[ "${DRY_RUN:-0}" == "1" ]] && DRY_RUN_FLAG="--dry-run"

REMOVE_FLAG=""
[[ "${REMOVE_REMOTE:-0}" == "1" ]] && REMOVE_FLAG="--remove-source-files"

LOCAL_REPO_ABS="$(cd "$LOCAL_REPO" && pwd)"

echo "==============================================================="
echo "=== pull_from_hpc: sliders + outputs (incrementale)         ==="
echo "==============================================================="

TOTAL=0

# Helper: conta SOLO i file effettivamente trasferiti (righe `>f...` o `<f...`),
# escludendo le righe di directory (`.d...` o `cd...`) che sono solo timestamp updates.
_count_files() {
    local out="$1"
    if [ -z "$out" ]; then echo 0; return; fi
    printf '%s\n' "$out" | grep -cE '^[<>]f' 2>/dev/null || echo 0
}

# --- 1. Sliders trainati (sdxl + flux) ---
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
        echo "[$(date +%H:%M:%S)] $arch sliders ($n file):"
        printf '%s\n' "$out" | grep -E '^[<>]f' | sed 's/^/    /'
        TOTAL=$((TOTAL + n))
    fi
done

# --- 2. Outputs immagini di tutti i task (sdxl + flux) ---
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
        echo "[$(date +%H:%M:%S)] $arch outputs ($n file):"
        printf '%s\n' "$out" | grep -E '^[<>]f' | sed 's/^/    /'
        TOTAL=$((TOTAL + n))
    fi
done

echo ""
if [ "$TOTAL" -eq 0 ]; then
    echo "[done] Tutto già aggiornato, niente da scaricare."
else
    echo "[done] $TOTAL file scaricati."
fi
[[ -n "$REMOVE_FLAG" ]] && echo "[note] file sorgente rimossi da HPC."
