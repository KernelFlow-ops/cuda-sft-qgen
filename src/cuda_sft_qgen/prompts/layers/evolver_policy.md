You are evolving a seed into a NEW CUDA exam question. Change the training structure; do not synonym-rewrite.

Seed (do not copy its sentence structure):
{{seed_question}}

Stay on this leaf claim:
{{l3_claim}}

This rewrite MUST use this ask-shape:
{{angle}}

Apply exactly ONE primary operator. Name it in `tests_claim` and in the JSON `operator` field.

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
   - Only when leaf kind is profiling/mechanism/failure, not syntax-only API trivia

6. **compare** — two architectures, APIs, or layouts; one falsifiable difference
   - GOOD: "sm_80 cp.async vs sm_90 TMA: which launch is legal for this tensormap?"
   - BAD: "compare Hopper and Ampere" with no concrete constraint
   - MUST: name both sides and the observable that differs

7. **counterfactual** — flip one premise and ask what breaks
   - GOOD: "same kernel, but drop __syncthreads: which race is now possible?"
   - BAD: "what if we used a different GPU?"
   - MUST: the flipped premise is explicit and the failure is unique

Match the operator to the leaf kind. Do not apply profiler to a syntax-only API leaf.

Stay answerable and on-leaf. Off-leaf (forbidden): TMA claim + cudaMalloc stem; bank-conflict claim + global-only stem.

The evolved stem must stay internally consistent: code, counts, masks, and `tests_claim` agree; exactly one correct option when a unique answer is requested; no leaked key in format examples.

This layer does not change the JSON schema. Still return `question`, `tests_claim`, `language`, `operator`.
