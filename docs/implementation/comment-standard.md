# Comment standard

Code comments in this repository follow Google-style docstrings in English.
Prompt files keep a versioned contract header. Comments explain constraints,
not the next line of code.

## Module docstring

Every public module starts with 3–8 lines covering:

1. What the module is for.
2. Invariants callers must not violate.
3. What this module is **not** responsible for.

Example (`graph/nodes/chair` conceptually):

```python
"""Independent quality review for one candidate question.

Accept requires every check to pass, a private reference answer, and
assessed metadata. Uncertainty and call failures stay pending and never
default to accept.

This module does not plan briefs or write questions.
"""
```

## Public functions and methods

One-sentence purpose. Add `Args` / `Returns` / `Raises` when the contract is
not obvious from types.

```python
def run_pipeline(settings: Settings, *, rebuild_taxonomy: bool = False) -> dict:
    """Run generate+review until target, iteration limit, or pending-only stop.

    Args:
        settings: Loaded runtime settings.
        rebuild_taxonomy: If True, Architect enqueues leaf expansion first.

    Returns:
        Final pipeline state with accepted/rejected/pending counts.

    Raises:
        RuntimeError: If another pipeline holds ``data_dir/pipeline.lock``.
    """
```

Trivial getters, argparse `set_defaults`, and one-line wrappers stay one
sentence or omit a docstring when the name is the contract.

## Inline comments

Write **why**, never **what**:

- Thresholds, backpressure, and backoff (`# leftover n=1 writer batches stall review`).
- Why a hardware/leaf mismatch is a warning rather than a hard reject.
- Why JSONL is a projection and SQLite is authoritative.

Do not restate `return 0`, `self.x = x`, or an `assert` message.

## Prompt files

Markdown prompts start with:

- Role (and what the role is not).
- Required input fields.
- Output JSON schema.
- Hard bans.
- A version tag (`QUALITY_REVIEW_V1`, `WRITER_PROMPT_V2`).

Keep the runtime loader and the file in sync. If `load_prompt("writer")` is
not called, the file must say so or be deleted.

## Data contracts

Pydantic models and TypedDict fields get a trailing comment only when the
constraint is not in the type (`pending: list[dict]  # mid-review candidates`).

## Tests

The test function docstring states the behavior under test. Do not comment
each `assert`.

## Language

- Source comments and docstrings: English.
- User-facing README / CLI help: Chinese is fine.
- Do not mix Chinese prose into Python docstrings.
