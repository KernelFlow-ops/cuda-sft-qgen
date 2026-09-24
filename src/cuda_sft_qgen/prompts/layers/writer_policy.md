## Coverage intent for this item: {{breadth_or_depth_signal}}
{{breadth_or_depth_guidance}}
- BREADTH (leaf has < 3 accepted items): attack a facet of the claim not yet covered — a different constraint class,
  failure path, observable, or scenario. Changing numbers on an existing ask-shape is NOT breadth.
- DEPTH (leaf is covered): keep the facet but add ONE interacting factor (arch gate × alignment, occupancy × latency,
  barrier × async proxy, precision × scale) so a solver needs one more reasoning step. Longer text is NOT depth.

## Angle lock
Required ask-shape: {{angle}}
{{angle_card}}
Other angles for this type are contrast only; do not merge them into this stem.

## Leaf lock
Background claim (never quote): {{l3_claim}}
`tests_claim` must contain at least one strong token of this leaf: {{strong_token_examples}}
Off-leaf examples (auto-reject): TMA-alignment claim answered with a cudaMalloc stem; bank-conflict claim with a
global-only stem; cp.async claim answered with TMA.

## Distractor quality (choice formats)
Each wrong option must be what a specific misconception produces (naive formula, wrong architecture, confusing UB
with deadlock, row- vs column-major). Name that misconception in `draft_rationale`. No joke options.

## Internal consistency (hard)
Code, comments, options and prose agree on every number (lanes, masks, bytes, SM counts). `tests_claim` describes
the stem you wrote, including its concrete instance. Do not mix mma.sync / wgmma / tcgen05 families.
