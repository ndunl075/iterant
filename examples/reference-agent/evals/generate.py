"""Regenerate ``evals/reference-pack`` (see below).

Run: ``uv run python examples/reference-agent/evals/generate.py``.

The tasks are written here, by hand, for Iterant (Apache-2.0). Every expected value is computed
from an arithmetic expression or from the reference agent's knowledge base, never typed in.
Output is deterministic. Regenerating changes the pack hash: do it when authoring, never on a
pack that a batch has pinned.

Grading is parse-only (ARCHITECTURE.md invariant 8). The agent writes ``/out/answer.json``
(``{"answer": "<text>"}``). The task's ``run`` script, executed by the grader in a sandbox,
reads it as strict JSON text and extracts one canonical value with regular expressions; it never
executes or imports anything from ``/out``. The hidden ``check`` holds the expected value.
"""

from __future__ import annotations

import ast
import json
import operator
import shutil
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from pathlib import Path

from iterant.packs.loader import split_pack

ROOT = Path(__file__).resolve().parent / "reference-pack"
GRADER_DIGEST = "a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704"
GRADER_IMAGE = "docker.io/library/python@sha256:" + GRADER_DIGEST
HEREDOC = "ITERANT_PY"

# kb/facts.json of the reference agent (a test keeps the two in step).
FACTS = {
    "founded": "2024",
    "headquarters": "Lisbon",
    "employees": "42",
    "product": "Widget Pro",
    "widget_price_usd": "19.99",
}

HINT_NUMBER = " Reply in one short sentence that ends with the number."
HINT_FACT = " Reply with the value only."
HINT_YESNO = " Answer yes or no."

_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}


def exact(expr: str) -> Fraction:
    """Exact value of an arithmetic expression (numbers, + - * / // % and integer **)."""

    def ev(node: ast.expr) -> Fraction:
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return Fraction(repr(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            power = ev(node.right)
            if power.denominator != 1:
                raise ValueError("integer exponents only")
            return ev(node.left) ** int(power)
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            return _BIN[type(node.op)](ev(node.left), ev(node.right))
        raise ValueError(f"unsupported expression: {expr}")

    return ev(ast.parse(expr, mode="eval").body)


def number_text(value: Fraction) -> str:
    """The canonical text a ``run`` script prints for a number: 2 decimals, half up, trimmed."""
    d = (Decimal(value.numerator) / Decimal(value.denominator)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    text = f"{d:f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


# (question, expression). Easy: one step. Medium: a short word problem.
# Hard: big operands, powers and rounding, where the model should use the calculator.
EASY = [
    ("What is 17 + 26?", "17 + 26"),
    ("What is 9 * 8?", "9 * 8"),
    ("What is 144 / 12?", "144 / 12"),
    ("What is 1000 - 387?", "1000 - 387"),
    ("What is 15% of 80?", "0.15 * 80"),
    ("What is 23 * 4 + 6?", "23 * 4 + 6"),
    ("What is (12 + 8) * 3?", "(12 + 8) * 3"),
    ("What is 250 / 4?", "250 / 4"),
    ("What is 2 to the power of 10?", "2 ** 10"),
    ("What is the remainder when 100 is divided by 7?", "100 % 7"),
]
MEDIUM = [
    ("A shop sells pens at 1.25 each. How much do 36 pens cost?", "36 * 1.25"),
    ("A train travels at 84 km per hour for 3.5 hours. How many km does it cover?", "84 * 3.5"),
    ("Rent is 1450 per month. How much is that over 12 months?", "1450 * 12"),
    ("You buy 3 items at 19.99 each and pay 5 for shipping. What is the total?", "3 * 19.99 + 5"),
    ("A book has 240 pages and you read 15 pages a day. How many days does it take?", "240 / 15"),
    ("What is the average of 12, 18, 25 and 41?", "(12 + 18 + 25 + 41) / 4"),
    ("What is the area of a rectangle that is 17 wide and 23 long?", "17 * 23"),
    ("Convert 3.2 kilometres to metres.", "3.2 * 1000"),
    (
        "A coat costs 80. It is reduced by 15%, then 8% tax is added. What is the final price?",
        "80 * 0.85 * 1.08",
    ),
    ("A bill of 156.60 is split equally among 6 people. How much does each pay?", "156.60 / 6"),
    ("How many seconds are there in 3 days?", "3 * 24 * 60 * 60"),
]
HARD = [
    ("What is 48213 * 379?", "48213 * 379"),
    ("What is 7919 * 6271?", "7919 * 6271"),
    ("What is 123456789 * 987?", "123456789 * 987"),
    ("What is (23.5 * 17) - (12.25 * 8) + 404?", "(23.5 * 17) - (12.25 * 8) + 404"),
    ("What is 17 to the power of 5?", "17 ** 5"),
    ("What is 1000 divided by 7, rounded to 2 decimal places?", "1000 / 7"),
    (
        "1000 is invested at 5% interest compounded yearly for 10 years. "
        "What is the final amount, rounded to 2 decimal places?",
        "1000 * 1.05 ** 10",
    ),
    ("What is the sum of all whole numbers from 1 to 250?", "250 * 251 / 2"),
    ("What is the remainder when 98765 is divided by 97?", "98765 % 97"),
]
# Questions about the demo company that also need arithmetic.
FACT_MATH = [
    ("How many years pass from the company's founding to 2031?", "2031 - 2024"),
    ("The company hires 8 more people. How many employees does it then have?", "42 + 8"),
    ("Revenue is 12600. What is the revenue per employee?", "12600 / 42"),
    ("The company doubles its headcount. How many employees is that?", "42 * 2"),
    ("What do 5 units of the company's product cost?", "5 * 19.99"),
    ("What do 12 units of the product cost after a 10% discount?", "12 * 19.99 * 0.9"),
    ("How many whole units of the product can you buy with 200?", "200 // 19.99"),
    (
        "A customer buys 3 units of the product and 2 gadgets at 4.50 each. What is the total?",
        "3 * 19.99 + 2 * 4.50",
    ),
    ("If each employee costs 55000 a year, what is the yearly staff cost?", "42 * 55000"),
]
FACT = [
    ("Which city is the company headquartered in?", "headquarters"),
    ("Where are the company's headquarters?", "headquarters"),
    ("In which year was the company founded?", "founded"),
    ("When was the company founded?", "founded"),
    ("How many employees does the company have?", "employees"),
    ("What is the company's headcount?", "employees"),
    ("What is the name of the company's product?", "product"),
    ("Which product does the company sell?", "product"),
    ("How much does one unit of the product cost in USD?", "widget_price_usd"),
    ("What is the product's unit price in USD?", "widget_price_usd"),
]
# (question, expected verdict)
YES_NO = [
    ("Is the company headquartered in Lisbon?", "yes"),
    ("Is the company headquartered in Porto?", "no"),
    ("Does the company have more than 40 employees?", "yes"),
    ("Is the company's product priced above 20 USD?", "no"),
    ("Is 97 a prime number?", "yes"),
    ("Is 91 a prime number?", "no"),
    ("Was the company founded before 2020?", "no"),
    ("Is 1000000 divisible by 64?", "yes"),
]

HEADER = '''"""Reference-pack check (@KIND@): parse-only; answer.json is read as JSON text."""
import json
import re
import sys
from decimal import ROUND_HALF_UP, Decimal

MAX_BYTES = 65536


def refuse(_name):
    raise ValueError("non-finite number")


def load_answer():
    with open("answer.json", "rb") as handle:
        raw = handle.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("too large")
    data = json.loads(raw.decode("utf-8"), parse_constant=refuse)
    if not isinstance(data, dict) or not isinstance(data.get("answer"), str):
        raise ValueError("need an object with a string `answer`")
    return data["answer"]
'''

NUMBER_BODY = """
NUMBER = re.compile(r"(?<![\\w.])-?(?:\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.\\d+)?")


def extract(text):
    found = NUMBER.findall(text)
    if not found:
        raise ValueError("no number")
    d = Decimal(found[-1].replace(",", "")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    out = f"{d:f}"
    out = out.rstrip("0").rstrip(".") if "." in out else out
    return {"number": out}
"""

YESNO_BODY = """
VERDICT = re.compile(r"\\b(yes|no)\\b", re.IGNORECASE)


def extract(text):
    found = VERDICT.findall(text)
    if not found:
        raise ValueError("no verdict")
    return {"verdict": found[-1].lower()}
"""

FACT_BODY = """
VALUES = @VALUES@


def extract(text):
    found = []
    for value in VALUES:
        pattern = r"(?<![A-Za-z0-9.])" + re.escape(value) + r"(?!\\.?[A-Za-z0-9])"
        if re.search(pattern, text, re.IGNORECASE):
            found.append(value.lower())
    return {"facts": sorted(found)}
""".replace("@VALUES@", repr(sorted(FACTS.values())))

FOOTER = """

def main():
    try:
        out = extract(load_answer())
    except Exception:
        sys.exit(1)
    sys.stdout.write(json.dumps(out, sort_keys=True, separators=(",", ":")) + "\\n")


main()
"""


def run_script(kind: str, body: str) -> bytes:
    program = HEADER.replace("@KIND@", kind) + body + FOOTER
    assert f"\n{HEREDOC}\n" not in program
    text = f"exec python3 -I - <<'{HEREDOC}'\n{program}{HEREDOC}\n"
    return text.encode("utf-8")


RUNS = {
    "number": run_script("number", NUMBER_BODY),
    "verdict": run_script("yes/no", YESNO_BODY),
    "facts": run_script("fact", FACT_BODY),
}


def dump(obj: object) -> bytes:
    return (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode("ascii")


def tasks() -> list[tuple[str, list[str], str, str, object]]:
    """``(id, tags, question, run kind, expected)`` in a fixed order."""
    out: list[tuple[str, list[str], str, str, object]] = []
    for level, group in (("easy", EASY), ("medium", MEDIUM), ("hard", HARD)):
        for n, (question, expr) in enumerate(group, 1):
            out.append(
                (
                    f"arith-{level}-{n:02d}",
                    ["arithmetic", level],
                    question + HINT_NUMBER,
                    "number",
                    {"number": number_text(exact(expr))},
                )
            )
    for n, (question, expr) in enumerate(FACT_MATH, 1):
        out.append(
            (
                f"fact-math-{n:02d}",
                ["fact-math"],
                "Using the demo company's facts, " + question[0].lower() + question[1:],
                "number",
                {"number": number_text(exact(expr))},
            )
        )
    for n, (question, key) in enumerate(FACT, 1):
        out.append(
            (
                f"fact-{n:02d}",
                ["fact"],
                question + HINT_FACT,
                "facts",
                {"facts": [FACTS[key].lower()]},
            )
        )
    for n, (question, verdict) in enumerate(YES_NO, 1):
        out.append(
            (
                f"yes-no-{n:02d}",
                ["yes-no"],
                question + HINT_YESNO,
                "verdict",
                {"verdict": verdict},
            )
        )
    return out


def main(root: Path = ROOT) -> None:
    """Write the pack to ``root`` (replacing what is there)."""
    if root.exists():
        shutil.rmtree(root)
    (root / "tasks").mkdir(parents=True)
    (root / "pack.toml").write_bytes(
        f'name = "reference-qa"\nversion = "1"\ngrader_image = "{GRADER_IMAGE}"\n'.encode()
    )
    (root / "README.md").write_bytes(
        b"# reference-qa\n\n"
        b"Small question-answering pack for the reference agent (arithmetic, the demo\n"
        b"company's facts, yes/no). Written for Iterant and licensed Apache-2.0 like the rest\n"
        b"of the repo. Regenerate with `examples/reference-agent/evals/generate.py`.\n\n"
        b"The agent writes `/out/answer.json`. Each task's `run` script (executed by the\n"
        b"grader sandbox) reads it as strict JSON text and extracts one canonical value with\n"
        b"regular expressions; nothing the agent wrote is executed. The hidden `check` holds\n"
        b"the expected value.\n"
    )
    for task_id, tags, question, kind, expected in tasks():
        directory = root / "tasks" / task_id
        directory.mkdir()
        (directory / "input.json").write_bytes(dump({"question": question}))
        (directory / "run").write_bytes(RUNS[kind])
        (directory / "check").write_bytes(dump(expected))
        (directory / "task.toml").write_bytes(
            f"tags = [{', '.join(json.dumps(t) for t in tags)}]\n".encode()
        )
    report = split_pack(root)
    print(f"wrote {report.n_tasks} tasks: {dict(report.counts)}")


if __name__ == "__main__":
    main()
