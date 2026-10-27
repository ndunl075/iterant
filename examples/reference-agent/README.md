# Reference agent

A small question-answering agent used to demo Iterant and as a CI fixture. It is its
own project: its dependencies are in `requirements.txt`, not in Iterant's.

- `agent/loop.py`: the model/tool loop (editable by Iterant).
- `prompts/`, `tools/`: system prompt and tools (editable).
- `kb/facts.json`, `agent/__main__.py`, `Containerfile`, `iterant.toml`: frozen.
- `evals/reference-pack/`: the task pack (57 self-written tasks: arithmetic, facts about the
  demo company, yes/no). `evals/reference-pack/generate.py` regenerates it. The pack directory is always
  excluded from the agent's image and the proposer's view, whatever `.containerignore` says.

Input `/task/input.json` is `{"question": "..."}`; output is `/out/answer.json`
(`{"answer": "..."}`) plus `/out/transcript.json`. LLM traffic goes through
`ANTHROPIC_BASE_URL`.

## Model

The agent does not choose its model. `agent/loop.py` reads the model and effort from the
frozen `iterant.toml` shipped in the image, the same file the proxy enforces. The file is set
up for the free profile (a local model through Ollama, `llama3.1:8b`); change `name` to use
another local model. For the paid claude profile, write a new config:

    iterant init . --force --profile claude --pack evals/reference-pack \
        --genome "prompts/**" --genome "tools/**" --genome agent/loop.py

## Try it

    ollama pull llama3.1:8b
    cd examples/reference-agent && git init && git add -A && git commit -m init
    iterant estimate .          # podman by default; --backend firecracker needs Linux + /dev/kvm

The pack has 57 tasks, so the splits are proportional (`iterant estimate` says so).
