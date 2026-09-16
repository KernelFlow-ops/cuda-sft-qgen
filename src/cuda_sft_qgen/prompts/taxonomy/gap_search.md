Search for missing knowledge points on ONE L2. This is NOT web search. Do not browse, cite URLs, or invent non-NVIDIA APIs. You are the librarian, not a question writer.

Stay inside this L2 only. Do not create L1 or L2 nodes.

Target L2: {{target_l2_id}} {{target_l2_name}}
Domain: {{domain_id}} {{domain_name}}

Search queries to satisfy (treat as holes, not titles to copy):
{{search_queries}}

Underrepresented kinds / types / tokens:
{{underrepresented}}

Leaves with high question-reject rate (prefer new claims that close these holes rather than restating them):
{{high_reject_leaves_json}}

Reject-reason histogram:
{{reject_reason_hist_json}}

Missing strong-token notes (claims or questions lacked these CUDA tokens):
{{missing_token_notes}}

Existing leaves (do NOT restate or paraphrase):
{{existing_leaves_json}}

Facet MUST for this call:
{{facet_must}}

Hard cap: at most {{max_new_leaves}} leaves (never more than 20). If nothing distinct and on-L2, return {"new_leaves": []}.

Fill holes in this order:
1. Reject / leaf_mismatch / unanswerable patterns that a sharper claim would fix (new leaf, not a rewrite — rewrites belong to refine).
2. Missing tokens that belong on this L2 (TMA, mbarrier, Shape/Stride, named Nsight counters, sm_ gates).
3. Underrepresented kinds, especially formula (`=` or `→` or `=>`), failure (path + observable), feature (arch + hard constraint).
4. Queries above that are not already covered by existing claims.

Rules:
- Prefix every `l3_id` with `{{target_l2_id}}_` (snake_case).
- Copy `domain_id={{domain_id}}`, `l2_id={{target_l2_id}}`, `l2_name={{target_l2_name}}` verbatim.
- Mix kinds unless facet MUST pins one. Prefer claims a writer can test without a blog.
- Each leaf MAY include `"source": "gap_search"`.
- Do not emit `question`, `tests_claim`, or `steering_brief`.

Return JSON only (same leaf shape as architect.md):

{
  "new_leaves": [
    {
      "domain_id": "T07",
      "l2_id": "layout_algebra",
      "l2_name": "Layout 代数",
      "l3_id": "layout_algebra_offset_inner_product",
      "claim": "one sentence technical claim",
      "types": ["layout", "conceptual"],
      "difficulty": [3, 4],
      "kind": "formula",
      "hints": ["optional instance"],
      "source": "gap_search"
    }
  ]
}
