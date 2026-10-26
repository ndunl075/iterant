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

**Policy: advisory only.** `pack.toml` sets `[policy] agent_code_exec_policy = "advisory-forbidden"`: the pack asks the agent not to run code, because running the function would turn the benchmark into a test of the interpreter. **Nothing enforces this.** The sandbox cannot tell a script the agent wrote to solve the task from one it ran to execute the function, so a candidate that runs code is not blocked and not detected. The field is there so the gate and the report can say so; do not read it as a guarantee, and treat a high score from a candidate that has code-execution tools with that in mind.

**Grader image.** `pack.toml` pins `grader_image` to a digest (`docker.io/library/python@sha256:...`, python 3.12 slim). The grader runs `run` with that image's `python3 -I`, never with the candidate-built target image, so the target needs no `python3` and a candidate cannot change what the grader executes. The pin is part of the pack hash. `Grader.preflight()` runs `python3 -V` in the image and fails as an infra error if it cannot. To move to a newer image, edit `GRADER_IMAGE` in `iterant/packs/cruxeval.py` and `pack.toml`; the pack hash changes.

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
