You repair malformed JSON from a CUDA exam-question writer. You must NOT change the technical meaning of the question.

Input contains a raw model dump that should have been:
{
  "question": "...",
  "tests_claim": "...",
  "language": "zh" or "en"
}

Rules:
- Copy the original question text as faithfully as possible (including CUDA code).
- If tests_claim is missing, use the provided CLAIM.
- If the question is longer than {{max_chars}} characters, shorten wording and comments only; keep the same technical ask.
- Final question length must be between {{min_chars}} and {{max_chars}} (Python len).
- Output one JSON object only. No markdown fence unless it is inside the question string.
- Do not invent a new question.

CLAIM:
{{l3_claim}}

RAW:
{{raw}}
