"""Review nodes: embedding score, independent chair audit, transactional archive.

Chair failures and uncertain checks stay pending. Dedup among accepted
survivors runs after quality review so a gray neighbor cannot skip the chair.

Implementations live in review_score, review_chair, and review_archive.
"""

from cuda_sft_qgen.graph.nodes.review_archive import archive_node
from cuda_sft_qgen.graph.nodes.review_chair import (
    _METADATA_KEYS,
    _defer_review,
    _enhanced_steering,
    _parallel_chair_decision,
    _parallel_chair_decisions,
    _curate_select,
    render_chair_prompt,
    review_chair_node,
)
from cuda_sft_qgen.graph.nodes.review_score import _question_fingerprint, review_score_node
from cuda_sft_qgen.runtime import get_runtime  # noqa: F401  — tests patch this barrel
