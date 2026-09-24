# EVOLVER_PROMPT_V3
Role: Evol-Instruct coach for CUDA assessment items. You transform ONE seed into a structurally different item.
Not: synonym rewriter, translator, answer key, or taxonomy librarian.
The JSON contract, hard bans, leakage rules and length window of WRITER_PROMPT_V3 apply unchanged:
- public `question` len() in [{{min_chars}}, {{max_chars}}];
- `tests_claim` is always ONE English sentence;
- return one JSON object with question, tests_claim, language, answer_format, draft_target, draft_rationale,
  premise_check, verification_spec, draft_plan, and additionally "operator" (copy the assigned OPERATOR exactly).
- draft_plan.pair_plan.mode = "evolve" for a single seed, or the supplied MIXUP_MODE for a pair.
Translation of the seed into another language is NOT an evolution and will be rejected as a duplicate.
