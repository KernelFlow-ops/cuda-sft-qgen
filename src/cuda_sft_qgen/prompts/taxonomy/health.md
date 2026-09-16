Analyze whether this CUDA/CUTLASS knowledge tree is reasonable as an exam-claim catalog. You are the librarian, not a writer and not the review chair.

Do NOT emit `new_leaves`. Do NOT rewrite claims. Do NOT invent L1/L2 nodes. Flag problems; send rewrites to `refine_ids` and holes to `search_queries` / `expansion_priority`.

Health summary (coverage, kind mix, short/generic claims, reject rates):
{{health_summary_json}}

Sample leaves (judge placement, answerability, kind vs wording):
{{sample_leaves_json}}

Reject-reason histogram from generated questions (use as a signal of unanswerable or mismatched leaves):
{{reject_reason_hist_json}}

What to look for:
- **misplaced**: claim belongs under a different existing L2 (e.g. TMA 128B alignment sitting under programming-model qualifiers instead of T08.tma / T07 layout). `suggested_l2` MUST be an existing `Txx.l2_id` key, not a new name.
- **unanswerable**: glossary, marketing, title-only, missing `=`/`→`/`=>` on a formula, missing arch+constraint on a feature, or a claim no writer can turn into a checkable item.
- **kind_mismatch**: wording is clearly another kind (formula identity declared as api; arch gate declared as mechanism).
- **expansion_priority**: existing L2 keys to grow first. Empty L2 > sparse L2 > L2 whose leaves have high reject / leaf_mismatch. Format `T07.layout_algebra`. Cap 12.
- **search_queries**: short CUDA-technical phrases for a later gap_search call (NO web, NO URLs). Example: `Hopper TMA expect_tx bytes vs mbarrier wait`. Cap 10.
- **refine_ids**: existing `l3_id` values that need a tighter falsifiable claim. Cap 20.
- **health_notes**: terse diagnostics (imbalance, promo language, kind walls, high leaf_mismatch). Cap 8.

Return JSON only (no markdown fence):

{
  "health_notes": ["..."],
  "misplaced": [{"l3_id": "...", "current_l2": "...", "suggested_l2": "...", "reason": "..."}],
  "unanswerable": [{"l3_id": "...", "reason": "..."}],
  "kind_mismatch": [{"l3_id": "...", "claim_kind": "formula", "declared_kind": "api"}],
  "expansion_priority": ["T07.layout_algebra"],
  "search_queries": ["..."],
  "refine_ids": ["..."]
}

Use empty arrays when a section has no findings. Every `l3_id` you cite MUST appear in the sample or summary payload — do not invent ids.
