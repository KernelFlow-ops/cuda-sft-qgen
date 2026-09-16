Runtime note: StructuredPrompt loads `layers/evolver_policy.md` as a runtime policy layer.
This file is the evolver role contract; JSON schema is unchanged.

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

1. **deepen** — add a numeric/alignment/occupancy constraint that changes the answer
   - GOOD: "stride must be divisible by 128" → "stride=256, shape=(4,8,16), alignment=128: legal?"
   - BAD: just changing stride=16 to stride=32 without structural consequence
   - MUST: question contains a falsifiable numeric constraint
   
2. **re-arch** — move Ampere constraint onto Hopper/Blackwell (TMA, cluster, TMEM) or vice versa
   - GOOD: "__syncthreads barrier" → "cluster-level mbarrier arrive/wait for Hopper"
   - BAD: simply stating "on Hopper" without using Hopper-specific features
   - MUST: question names the target architecture AND uses arch-specific API/mechanism
   
3. **inject-bug** — turn it into a short wrong kernel/layout and ask what fails
   - GOOD: working code → missing __syncthreads → "what race condition occurs?"
   - BAD: "this code has a bug, describe all possible issues"
   - MUST: question shows code snippet AND asks for observable consequence
   
4. **layout-eq** — rewrite as CuTe Shape/Stride/composition algebra
   - GOOD: "access pattern A[i][j]" → "Layout<Shape<_4,_8>, Stride<_8,_1>>, what's offset(2,3)?"
   - BAD: just mentioning "layout" without concrete Shape/Stride
   - MUST: question contains concrete Shape/Stride expressions
   
5. **profiler** — ask which Nsight Compute counter would move and in which direction
   - GOOD: "add __ldg loads" → "which counter: l1tex__t_sectors_pipe_lsu_mem_global_op_ld?"
   - BAD: "how would you profile this?" without specific metrics
   - MUST: question names specific Nsight Compute counter or metric category

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
