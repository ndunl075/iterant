"""Iterant's editing agent (§3 PROPOSE, §5). Implements ``iterant.contracts.Proposer``.

Its prompt (``prompt.md``) and genome (``genome.toml``) live beside the code so research mode
(§8) can evolve them.
"""

from iterant.proposer.agent import EditingProposer

__all__ = ["EditingProposer"]
