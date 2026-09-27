#!/usr/bin/env bash
# Unified verification entry (fix/03_TEST_CHAIN.md §1). Run from the repo root. All model-dependent steps use the REAL
# API from .env — there is no fake mode.
#   bash scripts/verify_all.sh --level unit|llm-smoke|wave1|wave2|wave3|eval
set -uo pipefail
LEVEL="unit"
while [[ $# -gt 0 ]]; do case "$1" in --level) LEVEL="$2"; shift 2;; *) echo "unknown arg $1"; exit 2;; esac; done
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
export QGEN_EVAL_OUT="${QGEN_EVAL_OUT:-$(mktemp -d)}"
FAIL=0
run() { echo "=== $*"; if ! "$@"; then echo "!!! FAILED: $*"; FAIL=1; fi; }
PYT="python -m pytest -q -p no:cacheprovider"

pure()  { run python -m compileall -q src/cuda_sft_qgen
          run python "$SCRIPTS/check_prompt_render.py"
          run $PYT tests/fix_v2/unit tests/fix_v2/component tests/fix_v2/e2e -m "not llm and not gpu"; }
smoke() { run python "$SCRIPTS/check_env.py" --ping
          run $PYT tests/fix_v2/llm/test_fv2_smoke_llm.py; }

case "$LEVEL" in
  unit)      pure ;;
  llm-smoke) smoke ;;
  wave1)     pure; smoke ;;
  wave2)     pure; smoke
             run $PYT tests/fix_v2/llm -m "llm and not eval" \
                 --deselect tests/fix_v2/llm/test_fv2_knowledge_llm.py \
                 --deselect tests/fix_v2/llm/test_fv2_gist_dedup_llm.py \
                 --deselect tests/fix_v2/llm/test_fv2_pipeline_real.py
             run python "$SCRIPTS/check_planner.py" --n 2000 ;;
  wave3)     pure; smoke
             run $PYT tests/fix_v2/llm -m "llm and not eval"
             run python "$SCRIPTS/check_planner.py" --n 2000
             TMP="$(mktemp -d)"
             run timeout "${QGEN_CLI_TIMEOUT:-2400}" python -m cuda_sft_qgen generate --target 10 --data-dir "$TMP/run"
             run python -c 'import json,sys; state=json.load(open(sys.argv[1], encoding="utf-8")); assert int(state.get("accepted_count", 0)) >= int(sys.argv[2]), state' "$TMP/run/run_state.json" 10
             run python -m cuda_sft_qgen stats --data-dir "$TMP/run"
             run python -m cuda_sft_qgen knowledge stats --data-dir "$TMP/run" ;;
  eval)      pure; smoke
             QGEN_CHAIR_STRICT=1 QGEN_CHAIR_REPEATS=3 run $PYT tests/fix_v2/llm -m "llm"
             run python -m cuda_sft_qgen review-benchmark --output "$QGEN_EVAL_OUT/review_bench.json"
             run bash "$SCRIPTS/baseline.sh" "$QGEN_EVAL_OUT/v2_run" 200 ;;
  *) echo "unknown level $LEVEL"; exit 2 ;;
esac
echo "=== outputs in $QGEN_EVAL_OUT"
echo "=== verify_all level=$LEVEL result=$([[ $FAIL == 0 ]] && echo PASS || echo FAIL)"
exit $FAIL
