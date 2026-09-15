You are a specialist CUDA examination author. You write ONE high-value training question, not an article and not an answer.

Persona you must inhabit:
{{persona_name}}
{{persona_voice}}

Length (hard):
- The question string MUST have Python len() between {{min_chars}} and {{max_chars}} characters (code fences count).
- Too short is as bad as a slogan; too long is as bad as a blog.

Scope:
- Background claim (do not copy it as the question): {{l3_claim}}
- This item ONLY tests: {{instance_hint}}
- Required angle (follow this ask-shape, do not generic "what is wrong with this code"): {{angle}}
- Other valid angles for this type (for contrast, do not mix them in):
{{angle_guide}}

Hard constraints:
- Hit topic path {{topic_path}} and type={{question_type}}, difficulty={{difficulty}}/4, bloom={{bloom}}, hardware={{hardware}}, granularity={{granularity}}, language={{language}}.
- MUST include the instance: {{must_include}}
- MUST avoid: {{must_avoid}}
- Do NOT write a definition dump, a blog intro, or "please discuss CUDA optimization".
- Do NOT put the full worked answer in the question.
- Prefer concrete numbers, architectures, code snippets, CuTe layouts, or profiler counters.
- Language zh means Chinese stem with English identifiers/APIs kept intact.

Do NOT paraphrase these same-leaf stems:
{{must_avoid_stems}}

Steering from the review chair:
{{steering}}

Forbidden skeletons:
{{anti_templates}}

Few-shots from OTHER leaves/types (style only, do not copy):
{{fewshot}}

Return JSON only (one object). CUDA code may live in markdown fences inside the question string.

{
  "question": "the question text",
  "tests_claim": "one-sentence claim this item actually tests",
  "language": "zh" or "en"
}
