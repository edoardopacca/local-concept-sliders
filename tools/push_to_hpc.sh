#!/usr/bin/env bash
# =============================================================================
# push_to_hpc.sh
# =============================================================================
# Pusha tutti gli .slurm dal Mac a HPC (incrementale: solo file nuovi o
# modificati). Mantiene la stessa struttura della repo: ogni .slurm finisce
# nella stessa cartella relativa su HPC.
#
# Copre: tutti i jobs/{old,new,test}_slurm/*.slurm di sdxl/ e flux/.
#
# Uso:
#   ./tools/push_to_hpc.sh                # pusha tutti gli slurm (default)
#   ./tools/push_to_hpc.sh new            # solo new_slurm/
#   ./tools/push_to_hpc.sh test           # solo test_slurm/
#   ./tools/push_to_hpc.sh old            # solo old_slurm/
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
case "$FILTER" in
    all)   ;;
    new|test|old) ;;
    *)     echo "Filtro invalido: $FILTER (usa: all|new|test|old)"; exit 1 ;;
esac

echo "==============================================================="
echo "=== push_to_hpc: .slurm $FILTER → HPC (incrementale)        ==="
echo "==============================================================="

# find compatibile con macOS (BSD find) e Linux (GNU find)
FILES=()
if [[ "$FILTER" == "all" ]]; then
    while IFS= read -r f; do
        FILES+=( "$f" )
    done < <(find . \
        \( -path '*/jobs/old_slurm/*.slurm' \
        -o -path '*/jobs/new_slurm/*.slurm' \
        -o -path '*/jobs/test_slurm/*.slurm' \) \
        -type f 2>/dev/null | sed 's|^\./||')
else
    while IFS= read -r f; do
        FILES+=( "$f" )
    done < <(find . -path "*/jobs/${FILTER}_slurm/*.slurm" -type f 2>/dev/null | sed 's|^\./||')
fi

if [ ${#FILES[@]} -eq 0 ]; then
    echo "[warn] nessun .slurm trovato per filtro '$FILTER'"
    exit 0
fi

echo "[info] ${#FILES[@]} .slurm da considerare"
echo ""

# Estrai le directory uniche (compatibile bash 3.2 — niente associative arrays)
UNIQUE_DIRS=$(printf '%s\n' "${FILES[@]}" | while IFS= read -r f; do dirname "$f"; done | sort -u)

TOTAL=0
while IFS= read -r dir; do
    [ -z "$dir" ] && continue
    src="$LOCAL_REPO_ABS/$dir/"
    dst="${HPC_USER}@${HPC_HOST}:${HPC_REPO}/$dir/"
    ssh "${HPC_USER}@${HPC_HOST}" "mkdir -p \"${HPC_REPO}/$dir\"" 2>/dev/null
    out=$(rsync -ahi --include='*.slurm' --exclude='*' "$src" "$dst" 2>&1 | grep -E '^[<>ch.]' || true)
    n=$(echo -n "$out" | grep -c '^' 2>/dev/null || echo 0)
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
    echo "[done] $TOTAL .slurm pushati."
fi
