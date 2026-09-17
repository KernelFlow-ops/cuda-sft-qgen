"""LangGraph pipeline that generates a CUDA-knowledge SFT question set.

This package plans taxonomy-aligned briefs, writes candidate questions,
deduplicates them, and independently reviews each item. Only accepted
questions are exportable; uncertain or failed reviews stay pending.
"""

__version__ = "0.1.0"
