"""Command-line entry point for generate, review, knowledge, and taxonomy tools.

Subcommands live in sibling modules. ``load_settings`` is re-exported so tests
can patch ``cuda_sft_qgen.cli.load_settings``.
"""

from __future__ import annotations

from cuda_sft_qgen.cli.common import _json_print, _setup_logging, load_settings
from cuda_sft_qgen.cli.config_cmds import (
    cmd_estimate_cost,
    cmd_gate,
    cmd_review_benchmark,
    cmd_validate_config,
)
from cuda_sft_qgen.cli.generate import (
    cmd_benchmark,
    cmd_export,
    cmd_generate,
    cmd_probe_concurrency,
    cmd_resume,
    cmd_stats,
)
from cuda_sft_qgen.cli.knowledge import cmd_knowledge
from cuda_sft_qgen.cli.parser import build_parser
from cuda_sft_qgen.cli.taxonomy import (
    cmd_expand_taxonomy,
    cmd_manage_taxonomy,
    cmd_rebuild_taxonomy,
    cmd_taxonomy_expand,
    cmd_taxonomy_health,
)

__all__ = [
    "build_parser",
    "cmd_benchmark",
    "cmd_estimate_cost",
    "cmd_expand_taxonomy",
    "cmd_export",
    "cmd_gate",
    "cmd_generate",
    "cmd_knowledge",
    "cmd_manage_taxonomy",
    "cmd_probe_concurrency",
    "cmd_rebuild_taxonomy",
    "cmd_resume",
    "cmd_review_benchmark",
    "cmd_stats",
    "cmd_taxonomy_expand",
    "cmd_taxonomy_health",
    "cmd_validate_config",
    "load_settings",
    "main",
]


def main(argv: list[str] | None = None) -> None:
    """Parse argv, run the selected subcommand, and exit with its status code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(getattr(args, "verbose", False))
    code = args.func(args)
    raise SystemExit(code)
