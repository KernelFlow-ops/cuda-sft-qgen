#!/usr/bin/env bash
# Real-model small batch (L7) + export + diversity (L8). Usage: bash scripts/baseline.sh <data_dir> <target>
set -euo pipefail
DIR="$1"; N="${2:-200}"
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
python "$SCRIPTS/check_env.py" >/dev/null || { echo "BLOCKED: .env not usable (run check_env.py)"; exit 1; }
mkdir -p "$DIR"
START=$(date +%s)
python -m cuda_sft_qgen generate --target "$N" --data-dir "$DIR" 2>&1 | tee "$DIR/generate.log"
END=$(date +%s)
python -m cuda_sft_qgen export --data-dir "$DIR" --output "$DIR/export.jsonl"
python -m cuda_sft_qgen stats --data-dir "$DIR" > "$DIR/stats.json"
LEAVES=$(python -c "from cuda_sft_qgen.config import load_settings; from cuda_sft_qgen.taxonomy import load_taxonomy; print(len(load_taxonomy(load_settings().taxonomy_path).leaves))")
python "$SCRIPTS/eval_diversity.py" "$DIR/export.jsonl" --taxonomy-leaves "$LEAVES" > "$DIR/diversity.json"
python "$SCRIPTS/gate_compare.py" summarize "$DIR" --wall-seconds $((END-START)) > "$DIR/summary.json"
cat "$DIR/summary.json"
