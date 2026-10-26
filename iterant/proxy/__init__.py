"""LLM egress proxy: the only path from any sandbox to a model (§6, invariants 3-5).

Orchestrator usage::

    proxy = Proxy(ProxyConfig(policy=..., sink=archive_sink, prices=..., split_of=...))
    server = ProxyServer(proxy, host="127.0.0.1").start()
    token = proxy.issue_run_token(Role.TARGET, run_key)   # one per RunKey
    # sandbox env: ANTHROPIC_BASE_URL=<route to server.base_url>, ANTHROPIC_API_KEY=<token>
    ...
    usage = proxy.run_usage(run_key)                      # spend and refusals, from the proxy
    proxy.begin_generation(n)                             # per-generation caps reset here

Non-run roles (proposer, analyst, auditor) use ``issue_call_token``. ``train_content``
returns call content for train-split runs only. ``FakeLLM`` (``fake.py``) is the shared
recorded-fixture upstream for tests. The ``free`` profile's Anthropic-to-OpenAI
translation is documented in ``translate.py``.
"""

from iterant.proxy.config import (
    CallSink,
    InMemorySink,
    ModelPrice,
    ProxyConfig,
    target_cap,
)
from iterant.proxy.core import Admission, CallContent, LedgerError, Proxy, RejectedCall
from iterant.proxy.fake import FakeLLM, anthropic_message, openai_completion
from iterant.proxy.ledger import RunUsage, Usage
from iterant.proxy.rules import Rejection
from iterant.proxy.server import ProxyServer

__all__ = [
    "Admission",
    "CallContent",
    "CallSink",
    "FakeLLM",
    "InMemorySink",
    "LedgerError",
    "ModelPrice",
    "Proxy",
    "ProxyConfig",
    "ProxyServer",
    "RejectedCall",
    "Rejection",
    "RunUsage",
    "Usage",
    "anthropic_message",
    "openai_completion",
    "target_cap",
]
