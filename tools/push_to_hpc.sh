#!/usr/bin/env bash
# =============================================================================
# push_to_hpc.sh
# =============================================================================
# Pusha dal Mac a HPC (incrementale, solo file nuovi o modificati):
#   - tutti gli .slurm in <arch>/{tasks/<task>/,trained_sliders/training/}/jobs/{old,new,test}_slurm/
#   - tutti i .yaml in <arch>/trained_sliders/training/configs/
#   - tutti i .yaml in <arch>/trained_sliders/training/prompts/{old,new,test}_prompt/
#
# Mantiene la stessa struttura della repo: ogni file finisce nella stessa
# cartella relativa su HPC.
#
# Uso:
#   ./tools/push_to_hpc.sh                # default: TUTTO (slurm + configs + prompts)
#   ./tools/push_to_hpc.sh slurm          # solo .slurm (qualsiasi cartella jobs/*)
#   ./tools/push_to_hpc.sh new            # solo new_slurm/
#   ./tools/push_to_hpc.sh test           # solo test_slurm/
#   ./tools/push_to_hpc.sh old            # solo old_slurm/
#   ./tools/push_to_hpc.sh configs        # solo configs/*.yaml
#   ./tools/push_to_hpc.sh prompts        # solo prompts/*/*.yaml
#   ./tools/push_to_hpc.sh yaml           # configs + prompts
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$SCRIPT_DIR/pull_config.sh" ]; then
    echo "ERROR: tools/pull_config.sh non trovato. Crealo da pull_config.sh.example."
    exit 1
fi
source "$SCRIPT_DIR/pull_config.sh"

LOCAL_REPO_ABS="$(cd "$LOCAL_REPO" && pwd)"
cd "$LOCAL_REPO_ABS"

FILTER="${1:-all}"

# -----------------------------------------------------------------------------
# Determina cosa pushare (lista di file)
# -----------------------------------------------------------------------------
FILES=()

push_slurm() {
    local sub="$1"  # "old", "new", "test", o "" per tutti
    local pattern
    if [ -z "$sub" ]; then
        # Tutti gli slurm in qualsiasi sotto-cartella *_slurm/
        while IFS= read -r f; do FILES+=( "$f" ); done < <(find . \
            \( -path '*/jobs/old_slurm/*.slurm' \
            -o -path '*/jobs/new_slurm/*.slurm' \
            -o -path '*/jobs/test_slurm/*.slurm' \) \
            -type f 2>/dev/null | sed 's|^\./||')
    else
        while IFS= read -r f; do FILES+=( "$f" ); done < <(find . \
            -path "*/jobs/${sub}_slurm/*.slurm" \
            -type f 2>/dev/null | sed 's|^\./||')
    fi
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
        push_slurm ""
        push_configs
        push_prompts
        ;;
    slurm)   push_slurm "" ;;
    new)     push_slurm "new" ;;
    test)    push_slurm "test" ;;
    old)     push_slurm "old" ;;
    configs) push_configs ;;
    prompts) push_prompts ;;
    yaml)
        push_configs
        push_prompts
        ;;
    *)
        echo "Filtro invalido: $FILTER"
        echo "Usa: all | slurm | new | test | old | configs | prompts | yaml"
        exit 1
        ;;
esac

echo "==============================================================="
echo "=== push_to_hpc: $FILTER → HPC (incrementale)               ==="
echo "==============================================================="

if [ ${#FILES[@]} -eq 0 ]; then
    echo "[warn] nessun file trovato per filtro '$FILTER'"
    exit 0
fi

echo "[info] ${#FILES[@]} file da considerare"
echo ""

# Estrai le directory uniche (compatibile bash 3.2)
UNIQUE_DIRS=$(printf '%s\n' "${FILES[@]}" | while IFS= read -r f; do dirname "$f"; done | sort -u)

# Helper: conta SOLO file effettivamente trasferiti (`>f...` o `<f...`).
# Robusto a stringhe vuote (set -e safe).
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
        echo "[$(date +%H:%M:%S)] $dir/  →  $n file:"
        echo "$out" | sed 's/^/    /'
        TOTAL=$((TOTAL + n))
    fi
done <<< "$UNIQUE_DIRS"

echo ""
if [ "$TOTAL" -eq 0 ]; then
    echo "[done] Tutto già aggiornato, niente da pushare."
else
    echo "[done] $TOTAL file pushati."
fi
