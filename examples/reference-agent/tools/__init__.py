"""Tools the agent can call. Add a module here and register it below."""

from tools.calculate import SCHEMA as _CALC_SCHEMA
from tools.calculate import calculate
from tools.lookup import SCHEMA as _LOOKUP_SCHEMA
from tools.lookup import lookup_fact

SCHEMAS = [_CALC_SCHEMA, _LOOKUP_SCHEMA]
TOOLS = {"calculate": calculate, "lookup_fact": lookup_fact}


def run_tool(name: str, args: dict) -> str:
    """Run one tool; errors come back as text so the model can recover."""
    fn = TOOLS.get(name)
    if fn is None:
        return f"error: unknown tool {name!r}"
    try:
        return str(fn(**args))
    except Exception as exc:  # tool errors are data for the model, not crashes
        return f"error: {exc}"
