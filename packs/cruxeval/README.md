# cruxeval-o: CRUXEval output prediction, parse-only

800 Python functions with an input; the agent predicts the return value.

## Grading (ARCHITECTURE invariant 8)
No agent-produced code is ever executed. The agent writes a Python literal to `answer.txt` in its
out dir. Each task's `run` script (identical for all 800, no expected values) reads it with
`python3 -I`, strips whitespace and one optional code fence, evaluates it with
`ast.literal_eval` only, and prints `{"answer": <canonical encoding>}`. The hidden `check` holds the
encoding of the dataset's expected output. Any failure (missing or oversized file, trailing junk,
too deep, unsupported type) exits non-zero and scores 0.0.

The encoding is type-tagged: equal iff Python `==`, except (stricter than upstream) `True` != `1`, tuple != list,
set != list and `1` != `"1"` (`1 == 1.0`, `-0.0 == 0`, NaN equals NaN). Its single source is
`iterant/packs/cruxeval_canon.py`; `run` embeds that file verbatim.

**Policy: the agent must not execute code for this pack** (`[policy] agent_code_exec = "forbidden"`
in `pack.toml`). Running the function would turn the benchmark into a test of the interpreter.
Nothing enforces this yet.

The target image needs `python3` (the grader's sandbox runs `run` with it). Without it every task silently scores 0.0 (exit 127); W2's startup check should verify it.

**Not comparable with published numbers.** Upstream CRUXEval-O runs `assert {output} == {gold}` on the model's text, so `1` passes for `True` and expressions like `'a'*3` pass. Here the answer must be a plain literal and types are kept apart, so scores are not directly comparable with published CRUXEval-O results.

## Source
- Repository: https://github.com/facebookresearch/cruxeval (MIT; see `LICENSE`, `NOTICE`)
- Commit: `190faf16d175b5847b0af05d937872b1fb395942`
- File: https://raw.githubusercontent.com/facebookresearch/cruxeval/190faf16d175b5847b0af05d937872b1fb395942/data/cruxeval.jsonl
- sha256 of `cruxeval.jsonl`: `8368b81047dc5014e4caf5a2f97604eff7644e0ecd7415e3ceeb184bbc2e0c96`

## Tags and splits
One tag per task from the length of the function source: `short` (< 100 characters), `medium`
(< 160), `long`. `splits.json` comes from `split_pack` with the default thresholds and seed 0.

## Regenerating
Download the file above, check its sha256, remove `pack.toml`, `splits.json` and `tasks/` from this
directory, then:

    uv run python -m iterant.packs.cruxeval cruxeval.jsonl packs/cruxeval --sha256 <sha256>

The output is deterministic. Regenerating changes the pack hash only if the converter changed.
