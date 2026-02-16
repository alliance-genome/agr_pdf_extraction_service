"""Compatibility export surface for consensus pipeline step helpers.

The implementation is split across focused modules:
- `consensus_micro_conflicts`
- `consensus_parsing_alignment`
- `consensus_classification_assembly`
- `consensus_hierarchy_qa`

This module re-exports symbols so existing imports remain stable.
"""

from __future__ import annotations

from app.services import consensus_classification_assembly as _consensus_classification_assembly
from app.services import consensus_hierarchy_qa as _consensus_hierarchy_qa
from app.services import consensus_micro_conflicts as _consensus_micro_conflicts
from app.services import consensus_parsing_alignment as _consensus_parsing_alignment


def _reexport_module_symbols(module) -> None:
    """Expose module symbols, including private helpers, for compatibility."""
    for name in dir(module):
        if name.startswith("__"):
            continue
        globals().setdefault(name, getattr(module, name))


_reexport_module_symbols(_consensus_micro_conflicts)
_reexport_module_symbols(_consensus_parsing_alignment)
_reexport_module_symbols(_consensus_classification_assembly)
_reexport_module_symbols(_consensus_hierarchy_qa)
