# reference-qa

Small question-answering pack for the reference agent (arithmetic, the demo
company's facts, yes/no). Written for Iterant and licensed Apache-2.0 like the rest
of the repo. Regenerate with `generate.py` in this directory.

The agent writes `/out/answer.json`. Each task's `run` script (executed by the
grader sandbox) reads it as strict JSON text and extracts one canonical value with
regular expressions; nothing the agent wrote is executed. The hidden `check` holds
the expected value.
