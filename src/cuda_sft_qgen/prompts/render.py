"""Single-pass ``{{name}}`` placeholder rendering for prompt templates.

Only the template text is scanned. Substituted values are inserted verbatim and
never re-scanned, so injected content (code skeletons such as ``{{N}, {N}}``,
steering stems, seeds, few-shot stems, knowledge quotes) may contain braces.
"""

from __future__ import annotations

import re
from typing import Mapping

_PLACEHOLDER = re.compile(r"\{\{([A-Za-z0-9_]+)\}\}")


def placeholders(template: str) -> set[str]:
    """Names of the ``{{name}}`` placeholders declared in ``template``."""
    return set(_PLACEHOLDER.findall(template))


def render(template: str, values: Mapping[str, object]) -> str:
    """Replace each declared placeholder with ``str(values[name])`` in one pass.

    A placeholder whose name is missing from ``values`` is left as written;
    callers compare :func:`placeholders` against their mapping to catch that.
    """
    def fill(match: re.Match[str]) -> str:
        name = match.group(1)
        return str(values[name]) if name in values else match.group(0)

    return _PLACEHOLDER.sub(fill, template)
