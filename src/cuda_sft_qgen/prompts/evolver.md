You are an Evol-Instruct coach for CUDA exam questions.

Take the seed question and produce a NEW question that is clearly different in training structure, not a synonym rewrite.

Seed question:
{{seed_question}}

Target claim / leaf:
{{l3_claim}}

This rewrite MUST use a different ask-shape:
{{angle}}
Instance to hit: {{instance_hint}}

Length (hard): Python len() of the question string in [{{min_chars}}, {{max_chars}}].

Apply ONE primary operator, named in tests_claim:
1. deepen — add a numeric/alignment/occupancy constraint that changes the answer
2. re-arch — move Ampere constraint onto Hopper/Blackwell (TMA, cluster, TMEM) or vice versa
3. inject-bug — turn it into a short wrong kernel/layout and ask what fails
4. layout-eq — rewrite as CuTe Shape/Stride/composition algebra
5. profiler — ask which Nsight Compute counter would move and in which direction

Constraints:
- type={{question_type}}, difficulty={{difficulty}}, hardware={{hardware}}, language={{language}}
- Must remain CUDA/NVIDIA GPU computing.
- Must stay answerable.
- Do not copy the seed sentence structure.
- Forbidden templates: {{anti_templates}}
- Do not paraphrase: {{must_avoid_stems}}
- Steering: {{steering}}

Return JSON only (one object). Code fences may appear inside the question string.

{
  "question": "...",
  "tests_claim": "...",
  "language": "zh" or "en",
  "operator": "deepen|re-arch|inject-bug|layout-eq|profiler"
}
