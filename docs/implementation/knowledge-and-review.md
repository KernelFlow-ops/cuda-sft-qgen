# Versioned Knowledge and Independent Review

Generation and review run concurrently with bounded queues. Every candidate receives
an independent model review, including candidates classified as unique by deduplication.
The reviewer receives the complete question, target claim, knowledge context, and
accepted neighboring questions. The seven checks cover correctness, answerability,
claim alignment, hardware compatibility, answer leakage, metadata, and duplication.

An accepted decision requires every check to pass, assessed metadata, and a private
reference answer. Invalid JSON, mismatched IDs, unavailable evidence, and model errors
leave the question pending. Delayed reviews use exponential backoff and eventually
remain exhausted for explicit investigation. They never become accepted by default.

## Storage and Migration

Each data directory contains independent `questions.sqlite3` and `knowledge.sqlite3`
databases. SQLite is authoritative. `accepted.jsonl`, `rejected.jsonl`, `pending.jsonl`,
and retry/state JSON files are compatibility projections and may be rebuilt.

On first use, existing JSONL records retain their IDs and are imported once. Original
files are copied to `.legacy` siblings before any projection changes. Terminal
records win over stale pending duplicates. Existing accepted questions retain their
historical state, but training export excludes records without an independent review.
Directly editing JSONL files after migration does not update the database.

Review decisions and counters commit in one transaction. Generation tasks have
durable IDs; retry tasks are leased and become recoverable after lease expiry.
Vector indexes are derived data, written atomically and reconciled against accepted
records at startup and before review. One generating process may use a data directory
at a time; background knowledge auditing uses SQLite transactions.

Internal reference answers and detailed evidence remain in the SQLite archive.
Public question JSONL removes those fields. `export` additionally requires a real
successful reviewer result and excludes withdrawn or pending questions. Fake model
results cannot be exported as verified training questions.

## Knowledge Lifecycle

The initial taxonomy keeps its domain, category, and leaf IDs. Imported claims are
`legacy_unreviewed`. Structurally valid new claims pass global deduplication and enter
as `provisional`; they are immediately available for generation, as requested.

The background worker retrieves official NVIDIA documents and stores document
snapshots, evidence quotes, revision history, audit outcomes, and typed relations.
Only quotes actually present in retrieved documents can support verification.
Insufficient evidence leaves a claim unverified; contradictions quarantine it.
Corrections create new revisions requiring another audit.

Relations support prerequisites, applicability, evidence, equivalence, contradiction,
and supersession. Question dependencies record the exact claim revisions used.
Knowledge changes withdraw affected accepted questions for review and produce
`corrections.jsonl`; previously exported datasets have manifests for tracking updates.

Question coverage and knowledge gaps are separate signals. Unasked knowledge goes
back to the planner; it does not by itself authorize adding more claims. Expansion
uses configured budgets, facet rotation, semantic deduplication, and cooldowns.

The worker schedules due audits while generation is running. For daily maintenance
when generation is stopped, invoke `knowledge audit` from an external scheduler.
Model-based source interpretation is still fallible; provenance enables inspection
and correction rather than proving all CUDA claims correct.

## Commands

```bash
cuda-sft-qgen generate --target 100 --data-dir data/new-run
cuda-sft-qgen resume --target 100 --data-dir data/new-run
cuda-sft-qgen resume --retry-reviews --data-dir data/new-run
cuda-sft-qgen knowledge stats --data-dir data/new-run --fake
cuda-sft-qgen knowledge query --claim glob_dev_host --data-dir data/new-run --fake
cuda-sft-qgen knowledge audit --limit 100 --data-dir data/new-run
cuda-sft-qgen knowledge impact --data-dir data/new-run --fake
cuda-sft-qgen knowledge export --output data/taxonomy-snapshot.yaml --data-dir data/new-run --fake
cuda-sft-qgen export --output data/training.jsonl --data-dir data/new-run
cuda-sft-qgen benchmark --target 100 --output data/benchmarks/run-100
cuda-sft-qgen review-benchmark --output data/benchmarks/review-report.json
```

`--retry-reviews` explicitly resets review retry budgets after the cause of uncertainty
or failure has been investigated. Persistently invalid knowledge must be corrected
before related questions can be accepted.

Configuration loading is shared by generation and validation. Precedence is explicit
arguments, environment, profile YAML, default YAML. `CUDA_SFT_PROFILE` selects the
profile, `REVIEW_MODEL` selects a separate review model, and `KNOWLEDGE_ENABLED=0`
disables background knowledge work. Defaults and limits live in `configs/default.yaml`.

## Measurement

The generation benchmark fixes a taxonomy snapshot, records model and concurrency
configuration, and writes throughput, leaf coverage, model calls, provider token
usage, pending counts, and a sample for inspection. Usage missing from the provider
remains unknown. Dollar cost is calculated only if explicit pricing is configured;
it is never inferred from sample token lengths. Accepted count alone is not quality.

The review benchmark freezes eight historical defects and four correct controls.
Expected labels are never sent to the reviewer. It stores full requests, responses,
latencies, checksums, and available usage in a private report. Defect misses, correct
question rejection, and pending outcomes are reported separately. This small
regression set does not establish a population error rate.
