You are evolving a seed into a NEW CUDA assessment item. The child must change the TRAINING STRUCTURE — what the
solver has to do — not the wording. A child whose answer can be obtained by copying the seed's answer is a failure.

Seed (read for mechanism only; do not reuse its sentences, code skeleton, or numbers):
{{seed_question}}

Leaf claim to stay on: {{l3_claim}}
Required ask-shape: {{angle}}
Operator assigned by the planner: {{operator}}   (you may NOT switch operator)

Operators and the dimension each MUST change:
1. deepen        — add one interacting constraint that changes the answer (arch gate, alignment, occupancy limit, precision).
2. re-arch       — move the scene to another architecture using that architecture's native mechanism
                   (sm_80 cp.async → sm_90 TMA+mbarrier → sm_100a tcgen05/TMEM); name both archs if comparing.
3. inject-bug    — show a short, realistic wrong kernel and ask for the observable consequence + minimal fix.
4. layout-eq     — restate the access pattern as CuTe Shape/Stride/composition and ask for offsets or a bijection check.
5. profiler      — ask which named Nsight Compute/Systems metric moves and by roughly how much, from given numbers.
6. compare       — two concrete alternatives, one falsifiable difference in an observable.
7. counterfactual— flip exactly ONE premise (explicitly) and ask what breaks.
8. format-shift  — same mechanism, different answer_format (e.g. numeric → code_fix, open → multi_choice with
                   misconception-driven distractors). The answer must be re-derived, not copied.
9. scenario-transfer — same mechanism inside a different workload (reduction → attention softmax, GEMM → KV-cache).
10. trace        — give a short program and ask for its exact output/state (deterministic code only).
11. decompose-reverse — give the result and ask for the missing parameter (e.g. given occupancy, find max regs).

Rules
- Match operator to leaf kind: profiler only for mechanism/failure/numeric/formula leaves; layout-eq only when an
  address mapping exists; re-arch only when the target architecture really has a counterpart mechanism.
- The child's `draft_target` must differ from the seed's answer.
- State in `draft_plan.construct` which dimension changed and why the seed's answer no longer applies.
- Same JSON contract as the writer, plus "operator": "{{operator}}".
