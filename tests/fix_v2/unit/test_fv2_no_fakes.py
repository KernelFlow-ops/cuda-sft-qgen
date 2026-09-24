"""Policy guard: no test under tests/fix_v2 may use FakeLLM, fake_llm=True, --fake, or hand-written model stubs."""
import re
from pathlib import Path

BANNED = [r"\bFakeLLM\s*\(", r"fake_llm\s*=\s*True", r"--fake\b", r"class\s+\w*(Stub|Fake|Mock)\w*\s*[\(:]",
          r"EMBEDDING_PROVIDER\s*=\s*['\"]fake", r"monkeypatch\.setattr\([^)]*complete"]


def test_no_fake_llm_usage():
    here = Path(__file__).resolve().parents[1]
    offenders = []
    for path in here.rglob("*.py"):
        if path.name in {"test_fv2_no_fakes.py", "conftest.py"}:
            continue
        text = path.read_text(encoding="utf-8")
        for pat in BANNED:
            for m in re.finditer(pat, text):
                offenders.append(f"{path.name}: {m.group(0)}")
    assert not offenders, offenders
