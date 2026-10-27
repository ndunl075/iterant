"""GATE stage (§3): static checks on a proposed diff before any sandbox time is spent.

Usage::

    auditor = ProxyAuditor(server.base_url, lambda: proxy.issue_call_token(Role.AUDITOR), model)
    gate = DiffGate(auditor, pack_root, protected=(config.evaluation.pack,))
    result = gate.check(proposal, repo=repo, target=config.target, pack=pack)
    result.passed, result.findings, result.warnings
"""

from iterant.gate.audit import (
    AuditorError,
    AuditParseError,
    AuditVerdict,
    ProxyAuditor,
)
from iterant.gate.diff import DiffError, FileDiff, parse_diff
from iterant.gate.gate import (
    Auditor,
    DiffGate,
    GateInfraError,
    GateReport,
    StaticChecks,
)

__all__ = [
    "AuditParseError",
    "Auditor",
    "AuditVerdict",
    "AuditorError",
    "DiffError",
    "DiffGate",
    "FileDiff",
    "GateInfraError",
    "GateReport",
    "ProxyAuditor",
    "StaticChecks",
    "parse_diff",
]
