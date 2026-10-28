#!/usr/bin/env bash
# Probe-only: install Ollama, pull the model, start the logging forwarder. Run from the repo root.
set -e
mkdir -p logs
curl -fsSL https://ollama.com/install.sh | sh
# Honest speed settings only (no change to model, context or sampling): keep the model resident,
# flash attention. Iterant issues one call at a time, so NUM_PARALLEL=1 is already all it can use.
export OLLAMA_KEEP_ALIVE=-1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_NUM_PARALLEL=1
(ollama serve > logs/ollama.log 2>&1 &)
for i in $(seq 1 30); do curl -fs http://127.0.0.1:11434/api/tags >/dev/null && break; sleep 2; done
ollama pull "$MODEL"
(python3 .github/scripts/logproxy.py logs/upstream.jsonl > logs/logproxy.log 2>&1 &)
sleep 2
ollama list
