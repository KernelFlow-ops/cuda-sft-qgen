Hard constraints (violate = unusable item):
- The question string MUST have Python len() in [{{min_chars}}, {{max_chars}}] (code fences count). Too short is a slogan; too long is a blog.
- MUST include the instance: {{must_include}}
- MUST avoid: {{must_avoid}}
- MUST NOT paraphrase these same-leaf stems:
{{must_avoid_stems}}
- MUST produce tests_claim as one falsifiable sentence that names the mechanism; do NOT copy l3_claim verbatim.
- Language {{language}}: Chinese stem keeps English identifiers/APIs intact.
- Hardware {{hardware}}: do not treat a Hopper/Blackwell-only mechanism as correct on Ampere (or vice versa).
- If HARDWARE is blackwell, write sm_100 / SM10x, never sm_90. Hopper is sm_90, Ampere is sm_80.
- When comparing two architectures (Hopper/Blackwell), name BOTH in the scene.
- Do NOT write a glossary, a blog intro, "请介绍/谈谈你对", or "畅所欲言".
- Do NOT hide the full worked answer in the question.
- Open-ended "how would you optimize" with no numbers, counters, or code is forbidden.
{{kind_constraint}}
- Hit topic path {{topic_path}} and type={{question_type}}, difficulty={{difficulty}}/4, bloom={{bloom}}, granularity={{granularity}}.
