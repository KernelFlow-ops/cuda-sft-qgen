# SALVAGE_PROMPT_V2
You repair MALFORMED JSON produced by a CUDA exam-question writer. You must NOT change the technical meaning.

The RAW dump below should have been one JSON object with these keys:
question, tests_claim, language, answer_format, draft_target, draft_rationale, premise_check,
verification_spec, draft_plan (and "operator" for evolver output).

Rules
1. Extract every key that is present in RAW, from any wrapping (markdown fence, trailing prose, truncated JSON).
   Remove fences that wrap the whole JSON; keep ``` fences that are INSIDE the question string.
2. Copy values verbatim. Do not paraphrase, do not fix "bugs" in code (they may be intentional), do not complete
   truncated code, do not invent missing private fields. A key absent from RAW stays absent from your output.
3. tests_claim: copy if present. If absent, write ONE English sentence describing only what the stem asks, sharing at
   least one CUDA identifier with the stem. Never copy or paraphrase the leaf claim below into tests_claim.
4. language: copy if present; otherwise "zh" when the stem contains CJK characters, else "en".
5. Keep the question text byte-for-byte after JSON decoding, even when it is too long or too short. The later
   validation stage decides whether to reject or regenerate it. The configured length window is
   [{{min_chars}}, {{max_chars}}]; do not edit the question to fit it.
6. Return exactly one JSON object and nothing else.

Leaf claim (context only, never copy): {{l3_claim}}

RAW:
{{raw}}
