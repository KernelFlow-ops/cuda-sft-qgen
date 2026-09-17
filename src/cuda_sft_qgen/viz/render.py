"""Render a knowledge-graph JSON document to a standalone HTML file."""
from __future__ import annotations

import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("graph.html")


def render_graph_html(graph: dict, output: Path | str, *, title: str = "CUDA knowledge graph") -> Path:
    """Inject ``graph`` JSON into ``graph.html`` and write ``output``."""
    output = Path(output)
    html = TEMPLATE.read_text(encoding="utf-8")
    payload = json.dumps(graph, ensure_ascii=False)
    html = html.replace("/*__KG_TITLE__*/", title.replace("</", "<\\/"))
    html = html.replace("/*__KG_DATA__*/", payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output
