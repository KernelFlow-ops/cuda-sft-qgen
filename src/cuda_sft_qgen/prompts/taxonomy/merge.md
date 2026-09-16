Adjudicate semantic near-duplicate L3 leaves. You are the librarian, not a question writer.

Input pairs (each has a, b, cosine). Cosine is a hint, not a verdict:
{{pairs_json}}

Emit exactly one decision per input pair. Use only `l3_id` values that appear in that pair. Do not invent ids. Do not rewrite claims.

Actions:
- `keep_both` — constraints actually differ (different arch gate, different formula/identity, different failure path, different API). Fill `keep_id` with the sharper leaf and `retire_id` with the other; neither will be dropped.
- `merge_into` — same mechanism and same constraint class; keep the more specific/falsifiable claim as `keep_id` and fold the other as `retire_id`.
- `retire` — true duplicates (same mechanism, same numbers/constraint, same arch, wording variants). `keep_id` survives; `retire_id` is dropped.

Default to `keep_both` when unsure. Retire only true duplicates. Do not merge Ampere-vs-Hopper gates, different equations, or bank-conflict vs global-coalescing just because cosine is high.

Prefer as `keep_id` the claim that is already falsifiable (has `=` / `→` / `=>` for formula, arch+constraint for feature, path+observable for failure).

Return JSON only (no markdown fence):

{
  "decisions": [
    {
      "keep_id": "...",
      "retire_id": "...",
      "action": "keep_both|merge_into|retire",
      "reason": "..."
    }
  ]
}

`reason` is one short sentence. No `new_leaves`, no `question`, no `tests_claim`.
