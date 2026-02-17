"""3-way global alignment package for consensus block matching."""

from app.services.alignment.arbitration import ArbitrationConfig, ArbitrationContext, choose_end_mode
from app.services.alignment.dp3 import DPResult, align_three_way_global
from app.services.alignment.partitioning import AnchorPartitionConfig, build_alignment_windows
from app.services.alignment.repair import repair_split_merge_columns
from app.services.alignment.scoring import ScoreConfig
from app.services.alignment.traceback import AlignmentColumn, traceback_columns
from app.services.alignment.triples import build_aligned_triples

__all__ = [
    "AlignmentColumn",
    "AnchorPartitionConfig",
    "ArbitrationConfig",
    "ArbitrationContext",
    "DPResult",
    "ScoreConfig",
    "align_three_way_global",
    "build_alignment_windows",
    "build_aligned_triples",
    "choose_end_mode",
    "repair_split_merge_columns",
    "traceback_columns",
]
