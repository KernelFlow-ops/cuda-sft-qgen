## Item specification (hard; violating any line makes the item unusable)
- TOPIC_PATH={{topic_path}}  TYPE={{question_type}}  KIND={{kind}}  DIFFICULTY={{difficulty}}/4  BLOOM={{bloom}}
- GRANULARITY={{granularity}}  LANGUAGE={{language}}  ANSWER_FORMAT={{answer_format}}  SCENARIO={{scenario}}
- ARCH_TARGET={{arch_target}} (hardware bucket {{hardware}}). Name the GPU / sm_ in the stem whenever the answer depends on it.
  Hopper = sm_90 (wgmma/setmaxnreg need sm_90a); data-center Blackwell = sm_100/sm_100a; GeForce Blackwell = sm_120 (no tcgen05/TMEM);
  Ampere = sm_80 (A100) / sm_86 / sm_87; Ada = sm_89. When two architectures are compared, name BOTH.
- Hardware facts you may state as givens (verified fact sheet; do not invent other limits):
{{hardware_facts}}
- Scene seed (build from it, do not quote it): {{instance_hint}}
- Must avoid: {{must_avoid}}
- Do NOT paraphrase these accepted stems on the same leaf (their full text is shown so you can check):
{{must_avoid_stems}}
- Already used on this leaf (pick something NOT in these lists): angles={{used_angles}}; scenarios={{used_scenarios}}; formats={{used_formats}}
{{kind_constraint}}
{{format_constraint}}

## Premise / answer separation (check before emitting)
| Allowed in stem (givens)                                   | Forbidden in stem (answer)                                  |
|------------------------------------------------------------|-------------------------------------------------------------|
| hardware limits the student cannot recall (reg file size)  | the rule being tested ("TMA requires 16B alignment")        |
| code, sizes, launch config, metric names to be interpreted | the numeric result, option letter, or corrected line        |
| definitions of non-standard notation you introduce         | "must mention" lists that enumerate the expected points     |
