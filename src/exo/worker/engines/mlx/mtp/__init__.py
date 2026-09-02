"""MTP speculative decoding — vendored from VQLab's `vqlab.mtp` package.

Provenance: verbatim copy of vqlab/mtp/{capture,caches,sampling,loop}.py
plus the two head modules (vqlab/mtp_head.py, vqlab/mtp_head_qwen35.py)
as of VQLab 2026-09-02; registry.py differs only in the vendored head
module paths. The loop's claims (1.56-1.80x measured, exact rejection
sampling at temperature, the committed-alignment scheme) are documented
and measured in that repo — fix bugs THERE first, then re-vendor, so the
single-box reference and this copy never diverge silently.

exo-specific glue lives in speculative.py, not in the vendored files.
"""

from .loop import MTPResponse, load_mtp_head, mtp_stream_generate
from .registry import FAMILIES, FamilySpec, register, resolve
from .speculative import maybe_mtp_head, mtp_responses

__all__ = [
    "MTPResponse",
    "load_mtp_head",
    "mtp_stream_generate",
    "FAMILIES",
    "FamilySpec",
    "register",
    "resolve",
    "maybe_mtp_head",
    "mtp_responses",
]
