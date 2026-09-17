Expand ONLY this L2. Do not add L1 domains. Do not add sibling L2 nodes. Do not write questions.

Domain: {{domain_id}} {{domain_name}}
L2: {{l2_id}} {{l2_name}}

Active facet: {{facet_id}}
Facet MUST: {{facet_must}}
Facet MUST NOT: {{facet_must_not}}
Facet example claim (style only — do not copy): {{facet_example}}

Hard cap: emit at most {{max_new_leaves}} leaves (never more than 20). If you cannot add a distinct falsifiable claim, return {"new_leaves": []}.

Existing leaves on this L2 (do NOT restate, retitle, or paraphrase these claims; do NOT reuse these l3_id values):
{{existing_leaves_json}}

Near-duplicate warnings (omit any candidate that matches these semantically):
{{near_dup_warnings}}

Pack coverage gaps (cover if missing; skip if already present in existing leaves):
{{gap_prompts}}

Rules for this call:
- Prefix every new `l3_id` with `{{l2_id}}_` (snake_case).
- Copy `domain_id`, `l2_id`, `l2_name` verbatim from this payload (`l2_name` may contain non-English tokens).
- Mix kinds (mechanism / api / formula / feature / failure / numeric) unless the facet MUST pins a kind; even then vary `types`.
- Respect the facet MUST / MUST NOT. The example claim is a quality bar, not a leaf to clone.
- formula claims MUST contain `=` or `→` or `=>`.
- feature claims MUST name sm_ / Hopper / Blackwell / Ampere / a CUDA version AND a hard constraint.
- Prefer fewer distinct leaves over near-duplicates of the existing list.

Return JSON only (no markdown fence):

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
      "hints": ["optional instance"]
    }
  ]
}

`hints` may be `[]`. Do not include `question`, `tests_claim`, `steering_brief`, or `source`.
