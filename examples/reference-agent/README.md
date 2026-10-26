# Reference agent

A small question-answering agent used to demo Iterant and as a CI fixture. It is its
own project: its dependencies are in `requirements.txt`, not in Iterant's.

- `agent/loop.py`: the model/tool loop (editable by Iterant).
- `prompts/`, `tools/`: system prompt and tools (editable).
- `kb/facts.json`, `agent/__main__.py`, `Containerfile`, `iterant.toml`: frozen.

Input `/task/input.json` is `{"question": "..."}`; output is `/out/answer.json`
(`{"answer": "..."}`) plus `/out/transcript.json`. LLM traffic goes through
`ANTHROPIC_BASE_URL`; the model is `claude-haiku-5-5`, effort medium.
