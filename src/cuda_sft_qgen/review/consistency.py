from __future__ import annotations

import re
from typing import Any

from cuda_sft_qgen.models import Question

# Scene-setting: "在 Hopper", "On an Ampere GPU", "compiled for Blackwell".
_SCENE_ARCH = re.compile(
    r"(?:在|on(?:\s+an?)?|for|compiled for|running on|executed on|执行于|运行于)\s+"
    r"(?:a(?:n)?\s+)?"
    r"(ampere|hopper|blackwell|a100|h100|b200|gh100|gb200)",
    re.IGNORECASE,
)
_PAREN_SM = re.compile(
    r"\b(ampere|hopper|blackwell)\s*\(\s*(?:sm_?\s*)?(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_ARCH_WORD = {
    "ampere": "ampere",
    "a100": "ampere",
    "ga100": "ampere",
    "hopper": "hopper",
    "h100": "hopper",
    "gh100": "hopper",
    "blackwell": "blackwell",
    "b200": "blackwell",
    "gb200": "blackwell",
}

_API = re.compile(
    r"\b(?:cuda[A-Z][A-Za-z0-9_]+|cu[A-Z][A-Za-z0-9_]+|NCCL_[A-Z0-9_]+|"
    r"nccl[A-Z][A-Za-z0-9_]+|CUB_[A-Z0-9_]+|cutlass[A-Za-z0-9_:]*)"
)
_INTRIN = re.compile(r"__[A-Za-z][A-Za-z0-9_]*__")
_PTX = re.compile(r"\b[a-z]{2,}\.[a-z]{2,}(?:\.[a-z0-9_]+)*")
_PTX_SKIP = {"e.g", "i.e", "vs.", "ex."}
_ID_PART_STOP = {
    "type",
    "types",
    "with",
    "from",
    "this",
    "that",
    "size",
    "mode",
    "kind",
    "core",
    "base",
    "main",
    "test",
    "item",
    "must",
}

_CHAIR_CONS = re.compile(
    r"leaf|hardware|hw_|mismatch|off_leaf|wrong.?arch|sieve_hw|sieve_off_leaf",
    re.IGNORECASE,
)


def _cc_arch(token: str) -> str | None:
    t = token.strip().lstrip("0") or "0"
    if t.startswith("10") or t.startswith("11"):
        return "blackwell"
    if t.startswith("9"):
        return "hopper"
    if t.startswith("8"):
        return "ampere"
    return None


def _canon_arch(word: str) -> str | None:
    return _ARCH_WORD.get(word.strip().lower())


def scene_arches(text: str) -> set[str]:
    found: set[str] = set()
    for m in _SCENE_ARCH.finditer(text or ""):
        arch = _canon_arch(m.group(1))
        if arch:
            found.add(arch)
    for m in _PAREN_SM.finditer(text or ""):
        arch = _canon_arch(m.group(1))
        if arch:
            found.add(arch)
    return found


def sm_name_conflicts(text: str) -> list[str]:
    """Arch name in 'Blackwell (SM90)' disagrees with the SM number."""
    hits: list[str] = []
    for m in _PAREN_SM.finditer(text or ""):
        named = _canon_arch(m.group(1))
        sm_arch = _cc_arch(m.group(2))
        if named and sm_arch and named != sm_arch:
            hits.append(f"{m.group(1)}(SM{m.group(2)})")
    return hits


def hardware_mismatch(question: Question) -> str | None:
    text = question.question or ""
    labeled = (question.hardware or "agnostic").strip().lower()
    conflicts = sm_name_conflicts(text)
    if conflicts:
        return "sieve_hw_mismatch"
    if labeled not in {"ampere", "hopper", "blackwell"}:
        return None
    scenes = scene_arches(text)
    if not scenes:
        return None
    if labeled in scenes:
        return None
    # Compare items may name two architectures; still require the labeled one
    # to appear if the stem sets a scene at all.
    if question.question_type == "compare" and len(scenes) >= 2:
        return None
    return "sieve_hw_mismatch"


def _strong_tokens(*texts: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    blob = "\n".join(t for t in texts if t)
    for rx in (_API, _INTRIN, _PTX):
        for m in rx.finditer(blob):
            tok = m.group(0)
            key = tok.lower().rstrip(".")
            if key in _PTX_SKIP or key in {"cuda", "cutlass"}:
                continue
            if len(key) < 6:
                continue
            if key not in seen:
                seen.add(key)
                out.append(tok)
    return out


def _id_parts(l3_id: str) -> list[str]:
    parts = []
    for p in re.split(r"[_\-.]+", l3_id or ""):
        pl = p.lower()
        if len(pl) < 4 or pl in _ID_PART_STOP:
            continue
        parts.append(pl)
    return parts


def _leaf_claim(question: Question) -> str:
    claim = (getattr(question, "l3_claim", None) or "").strip()
    if claim:
        return claim
    try:
        from cuda_sft_qgen.runtime import get_runtime

        rt = get_runtime()
        for leaf in rt.taxonomy.leaves:
            if leaf.l3_id == question.l3_id:
                return leaf.claim
    except Exception:
        return ""
    return ""


def _contains_token(haystack: str, token: str) -> bool:
    h = haystack.lower()
    t = token.lower()
    if t in h:
        return True
    # cudaIpcGetMemHandle vs cudaipcgetmemhandle already covered; dotted PTX
    return t.replace("_", "") in h.replace("_", "")


def leaf_mismatch(question: Question) -> str | None:
    claim = _leaf_claim(question)
    qtext = question.question or ""
    tests = question.tests_claim or ""
    hay = f"{qtext}\n{tests}"
    strong = _strong_tokens(claim, question.l3_id.replace("_", " "))
    if strong:
        if any(_contains_token(hay, tok) for tok in strong):
            return None
        # Claim APIs never appear in stem or tests_claim: wrong leaf.
        return "sieve_off_leaf"
    parts = _id_parts(question.l3_id)
    if not parts:
        return None
    if any(p in hay.lower() for p in parts):
        return None
    return "sieve_off_leaf"


def consistency_reason(question: Question) -> str | None:
    return hardware_mismatch(question) or leaf_mismatch(question)


def leaf_soft_suspect(question: Question) -> bool:
    """True when the stem may mash topics: claim APIs missing from tests_claim."""
    if consistency_reason(question):
        return True
    claim = _leaf_claim(question)
    strong = _strong_tokens(claim, (question.l3_id or "").replace("_", " "))
    if not strong:
        return False
    tests = question.tests_claim or ""
    return not any(_contains_token(tests, tok) for tok in strong)


def chair_may_reject_unique(reason: str | None) -> bool:
    if not reason:
        return False
    return bool(_CHAIR_CONS.search(reason))


def consistency_payload(question: Question) -> dict[str, Any]:
    return {
        "id": question.id,
        "l3_id": question.l3_id,
        "l3_claim": (getattr(question, "l3_claim", None) or "")[:240],
        "hardware": question.hardware,
        "tests_claim": (question.tests_claim or "")[:240],
        "question": (question.question or "")[:500],
        "rule_reason": consistency_reason(question),
    }
