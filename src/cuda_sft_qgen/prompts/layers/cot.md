## Construction procedure (reason privately; expose only the concise private fields of the output schema)
1. Analyze: restate CLAIM as a checkable proposition; list what the student must compute/decide to reach it.
   Decide which facts are GIVENS and which one is the ANSWER. KIND={{kind}}: {{cot_scene}}
2. Plan the scene: pick SCENARIO={{scenario}} and ANSWER_FORMAT={{answer_format}}; choose numbers/code so that
   (a) the answer is unique, (b) a common misconception yields a DIFFERENT wrong answer (a good distractor),
   (c) nothing in the scene equals the answer by inspection.
3. Construct the stem, then compute `draft_target` yourself from the stem only (not from CLAIM).
4. Adversarial self-check — reject and rebuild the draft if any line fails:
   - premise_check: every fact the target depends on appears in the stem (list them).
   - uniqueness: for choice formats, evaluate EVERY option under the same premises; exactly the keyed ones are correct.
   - code: read literally; branch conditions, loop bounds, masks, sizes and comments agree with the prose.
   - arithmetic: recompute with units; the result supports tests_claim (not its opposite).
   - leakage: no sentence states the rule, result, or fix; format hints contain no key.
   - hardware: every API/opcode exists on ARCH_TARGET; UB is described as UB.
   - novelty: the item differs from every must_avoid stem in mechanism facet, scene, or format — not just numbers.

## verification_spec rules
- scalar: single number/enum answer; `draft_target` contains ONLY that value (units fixed by the stem).
- open: explanations, multi-part answers, trade-offs; `rubric` = 2–5 atomic, independently checkable points.
- symbolic: bounded arithmetic `expression` in named variables with `variables` bound.
- compile/runtime: complete translation unit with headers and main(), deterministic stdout, `architecture` equal
  to ARCH_TARGET; never claim UB deterministically crashes; use open for UB semantics.
Choose ONE literal value for every enum. seed_analysis.domain_id = first TOPIC_PATH component.
pair_plan.mode: single (writer), evolve (single-seed evolver) or the supplied MIXUP_MODE.
{{cot_evolver}}
