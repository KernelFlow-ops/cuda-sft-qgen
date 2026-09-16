Rewrite weak L3 claims. You are the librarian, not a question writer.

Keep the technical meaning. Do NOT invent new `l3_id` values. Do NOT add leaves. Do NOT change kind unless the payload's `issues` explicitly say the declared kind is wrong — and even then, fix the claim wording to match the intended kind rather than silently switching topics.

Input leaves (each has l3_id, claim, kind, hints, issues):
{{leaves_json}}

For every input leaf, emit exactly one suggestion with the same `l3_id`.

Rewrite rules:
- Make the claim one falsifiable sentence (a grader can mark true/false).
- Keep NVIDIA CUDA/CUTLASS/CuTe/PTX/Nsight tokens already in the claim; do not swap in a different mechanism.
- kind=formula: `suggested_claim` MUST contain `=` or `→` or `=>` and name inputs → unique output.
- kind=feature: MUST name sm_ / Hopper / Blackwell / Ampere / a CUDA version AND a hard constraint.
- kind=failure: MUST name a path and an observable (deadlock, silent wrong, UB, hang).
- kind=mechanism: MUST give a hardware/execution reason, not a glossary.
- kind=api: MUST state legal vs illegal / parameter / return constraint.
- kind=numeric: MUST include numbers (or a fully bound symbol set) and a unique result.
- `suggested_hints`: 1–3 concrete instances (numbers, APIs, Shape/Stride, counters). If the input hints are already good, keep them.
- Echo residual `issues` you could not fully fix; use `[]` if the rewrite clears them.
- If the claim is already falsifiable, tighten wording only; do not change the mechanism.

BAD rewrites (forbidden):
- glossary: "CuTe Layout is a pairing of Shape and Stride."
- marketing: "Hopper TMA greatly improves copy performance."
- a different leaf: TMA alignment rewritten as cudaMalloc advice.

Return JSON only (no markdown fence):

{
  "suggestions": [
    {
      "l3_id": "...",
      "suggested_claim": "...",
      "suggested_hints": ["..."],
      "issues": ["..."]
    }
  ]
}

Suggestion count MUST equal input leaf count. No `new_leaves`, no `question`, no `tests_claim`.
