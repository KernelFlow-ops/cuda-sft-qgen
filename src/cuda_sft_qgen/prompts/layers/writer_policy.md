## Breadth vs Depth

- BREADTH: the leaf has few existing questions — hit a different facet of the claim (constraint class, failure path, or profiler metric). Do not synonym-rewrite a prior stem.
- DEPTH: the leaf already has coverage — go deeper on one facet with a harder numeric, alignment, mask, or architecture edge.
- For this item: {{breadth_or_depth_signal}} — {{breadth_or_depth_guidance}}
- Repeating the same ask-shape with different numbers is paraphrasing, not depth.

## Angle and variation

Required ask-shape: {{angle}}
Follow this angle only. Do not mix a second angle into the same stem.

If this same angle has already been used on this leaf, you MUST still follow {{angle}} but vary at least one of:
- technical constraint (prior stride=16 → try alignment=128 or mask=0xf)
- failure mode (prior deadlock → race or silent corruption)
- profiler metric (prior l1tex stall → dram throughput or warp occupancy)

## Leaf lock

Background claim (do not copy it as the question): {{l3_claim}}
Instance to hit: {{instance_hint}}

`tests_claim` MUST share at least one strong token with {{l3_claim}} or {{instance_hint}}.
Strong tokens for this item: {{strong_token_examples}}
Typical strong tokens: cuda* APIs, __* intrinsics, sm_XX, Hopper/Ampere/Blackwell, PTX ops, CuTe primitives, named Nsight counters.

Off-leaf (forbidden — the chair will reject):
- Claim is TMA alignment, question uses cudaMalloc host allocation.
- Claim is shared-memory bank conflicts, question uses only global memory.

This layer does not change the JSON schema. Still return one object with `question`, `tests_claim`, `language`.
