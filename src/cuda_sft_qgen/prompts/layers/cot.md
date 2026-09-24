Before emitting JSON, reason privately through these four stages. Keep each field concise;
do not expose a free-form reasoning transcript.

1. Analyze: identify the falsifiable mechanism, required instance, answer type,
   source seed IDs/domain, and reusable constraints from CLAIM + INSTANCE + KIND={{kind}}.
2. Plan: choose one concrete test scene ({{cot_scene}}), its premises, a unique
   observable/answer, and a variation that is not a paraphrase of a seed. For a
   MathMixup pair, retain the anchored claim and name the transferred constraint.
3. Construct: write the self-contained question and a provisional target answer
   that can be checked independently. Apply difficulty {{difficulty}} and follow
   ONLY "{{angle}}"; other angles are contrast, not mix-ins.
4. Self-check: verify claim alignment, units/counts/code order, hardware scope,
   answerability, answer leakage, uniqueness, and the verification contract.
   Reject your own draft internally if any check fails and construct a replacement.

Return one JSON object only. In addition to the public fields, include these
private fields for the reviewer (they are removed before JSONL export):
"draft_plan": {"seed_analysis": {"source_seed_ids": [], "domain_id": "...", "l3_ids": [], "shared_mechanism": "...", "reusable_constraints": [], "differences": []}, "pair_plan": {"mode": "single|evolve|original|decomposed|hybrid", "parent_ids": [], "retained_claim": "...", "transferred_constraint": "...", "construction": "...", "expected_difficulty": {{difficulty}}}, "construct": "short design summary", "self_check": ["claim_alignment", "answerability", "verification_contract"]},
"draft_target": "...",
"verification_spec": {"mode": "scalar|symbolic|open|compile|runtime", "answer_type": "number|enum|text|expression", "expected_behavior": "one concise expected answer/observable", "rubric": ["..."], "expression": "optional", "variables": {}, "code": "complete source when compile/runtime", "architecture": "sm_XX when compile/runtime", "expected_compile": "success|failure", "expected_diagnostic": "optional", "expected_stdout": "required for runtime", "sanitizer": "none|memcheck|racecheck", "toolkit_version": "optional"}.
The target and verification fields are private and never exported. `draft_target`
is the expected answer or observable, not a solution transcript. `tests_claim`
must remain one sentence a grader can mark true/false without a full solution.
Choose ONE literal value for every enum; never emit the vertical-bar alternatives.
Omit optional fields that do not apply. The public question must state every
premise needed by the oracle. Use scalar only for a single numerical result;
draft_target then contains ONLY that number, with units specified in the question.
For explanations, multiple requested results, or tradeoffs use open/text and a
short criterion list. A prose answer is not a scalar or a symbolic expression.
For symbolic, expression must be a bounded arithmetic expression in named
variables; variables supplies numeric values when appropriate.
For compile/runtime, code must include all headers, declarations and (for
runtime) main(), error checking and deterministic stdout. Architecture must
match the question's stated GPU, not a convenient available machine. Never
claim UB will deterministically deadlock or crash. If verification requires
reasoning about UB, use open with the semantic rule as its rubric.
seed_analysis.domain_id must be exactly the first TOPIC_PATH component;
use an empty source_seed_ids list when no compatible seed was supplied.
For pair_plan.mode select single for Writer, evolve for single-seed Evolver,
or the supplied MIXUP_MODE. Keep summaries concise (no private thought transcript).
{{cot_evolver}}
