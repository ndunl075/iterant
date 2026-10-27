"""Regenerate ``evals/reference-pack`` (see below).

Run: ``uv run python examples/reference-agent/evals/reference-pack/generate.py``.

The tasks are written here, by hand, for Iterant (Apache-2.0). Every expected value is computed
from an arithmetic expression or from the reference agent's knowledge base, never typed in.
This file lives inside the pack, so it is covered by the pack hash and by the hard exclusion
of the pack directory from every sandbox image and from the proposer's view: the answers it
computes can never reach the agent. Output is deterministic. Regenerating changes the pack
hash: do it when authoring, never on a pack that a batch has pinned.

Grading is parse-only (ARCHITECTURE.md invariant 8). The agent writes ``/out/answer.json``
(``{"answer": "<text>"}``). The task's ``run`` script, executed by the grader in a sandbox,
reads it as strict JSON text and extracts one canonical value with regular expressions; it never
executes or imports anything from ``/out``. The hidden ``check`` holds the expected value.
"""

from __future__ import annotations

import ast
import json
import operator
import random
import shutil
from collections.abc import Callable
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from pathlib import Path

from iterant.packs.loader import split_pack

ROOT = Path(__file__).resolve().parent  # this file lives inside the pack
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

# --- generated extensions -----------------------------------------------------------
# The hand-written lists above are the first tasks of each category; the rest are drawn from
# templates with a fixed seed, so the pack is byte-for-byte reproducible. Every expected value
# still comes from ``exact`` / ``FACTS``, never from a typed-in constant.
_RNG = random.Random(20240611)


def _money(cents: int) -> str:
    return f"{cents // 100}.{cents % 100:02d}"


def _fill(group: list[tuple[str, str]], target: int, make: Callable[[], tuple[str, str]]) -> None:
    """Append template draws to ``group`` until it has ``target`` unique questions."""
    seen = {q for q, _ in group}
    while len(group) < target:
        question, expr = make()
        if question not in seen:
            seen.add(question)
            group.append((question, expr))


def _easy() -> tuple[str, str]:
    r = _RNG
    kind = r.randrange(8)
    a, b = r.randrange(11, 99), r.randrange(11, 99)
    if kind == 0:
        return f"What is {a} + {b}?", f"{a} + {b}"
    if kind == 1:
        hi, lo = max(a, b) + 100, min(a, b)
        return f"What is {hi} - {lo}?", f"{hi} - {lo}"
    if kind == 2:
        x, y = r.randrange(3, 13), r.randrange(3, 13)
        return f"What is {x} * {y}?", f"{x} * {y}"
    if kind == 3:
        y, q = r.randrange(3, 13), r.randrange(4, 30)
        return f"What is {y * q} / {y}?", f"{y * q} / {y}"
    if kind == 4:
        p, n = r.choice((10, 20, 25, 50, 5)), r.randrange(2, 20) * 20
        return f"What is {p}% of {n}?", f"{p / 100} * {n}"
    if kind == 5:
        x, y, z = r.randrange(5, 30), r.randrange(3, 9), r.randrange(2, 40)
        return f"What is {x} * {y} + {z}?", f"{x} * {y} + {z}"
    if kind == 6:
        x, y, z = r.randrange(5, 30), r.randrange(5, 30), r.randrange(2, 9)
        return f"What is ({x} + {y}) * {z}?", f"({x} + {y}) * {z}"
    n, d = r.randrange(50, 500), r.randrange(3, 12)
    return f"What is the remainder when {n} is divided by {d}?", f"{n} % {d}"


def _medium() -> tuple[str, str]:
    r = _RNG
    kind = r.randrange(12)
    if kind == 0:
        n, c = r.randrange(12, 90), r.randrange(80, 900)
        q = f"A shop sells notebooks at {_money(c)} each. How much do {n} notebooks cost?"
        return q, f"{n} * {_money(c)}"
    if kind == 1:
        v, t = r.randrange(40, 130), r.randrange(2, 9) + r.choice((0, 0.5))
        q = f"A bus travels at {v} km per hour for {t} hours. How many km does it cover?"
        return q, f"{v} * {t}"
    if kind == 2:
        rent, m = r.randrange(600, 2400, 25), r.randrange(3, 18)
        return f"A flat costs {rent} per month. How much is that over {m} months?", f"{rent} * {m}"
    if kind == 3:
        n, c, ship = r.randrange(2, 9), r.randrange(200, 3000), r.randrange(2, 15)
        q = f"You buy {n} items at {_money(c)} each and pay {ship} for shipping. What is the total?"
        return q, f"{n} * {_money(c)} + {ship}"
    if kind == 4:
        d, per = r.randrange(5, 40), r.randrange(3, 25)
        q = (
            f"A report has {d * per} pages and you proofread {per} pages a day. "
            "How many days does it take?"
        )
        return q, f"{d * per} / {per}"
    if kind == 5:
        v = [r.randrange(5, 99) for _ in range(4)]
        v[0] += (4 - sum(v) % 4) % 4  # keep the mean a short decimal
        return (
            f"What is the average of {v[0]}, {v[1]}, {v[2]} and {v[3]}?",
            f"({v[0]} + {v[1]} + {v[2]} + {v[3]}) / 4",
        )
    if kind == 6:
        w, h = r.randrange(8, 60), r.randrange(8, 60)
        return f"What is the area of a rectangle that is {w} wide and {h} long?", f"{w} * {h}"
    if kind == 7:
        km = r.randrange(11, 99) / 10
        return f"Convert {km} kilometres to metres.", f"{km} * 1000"
    if kind == 8:
        price, off, tax = r.randrange(40, 400, 10), r.choice((10, 20, 25, 30)), r.choice((5, 8, 10))
        q = (
            f"A jacket costs {price}. It is reduced by {off}%, then {tax}% tax is added. "
            "What is the final price?"
        )
        return q, f"{price} * {(100 - off) / 100} * {(100 + tax) / 100}"
    if kind == 9:
        n, share = r.randrange(3, 9), r.randrange(500, 5000)
        q = (
            f"A bill of {_money(n * share)} is split equally among {n} people. "
            "How much does each pay?"
        )
        return q, f"{_money(n * share)} / {n}"
    if kind == 10:
        pay, cost = r.randrange(50, 200), r.randrange(500, 4900)
        q = f"You pay with {pay} for an item that costs {_money(cost)}. How much change is that?"
        return q, f"{pay} - {_money(cost)}"
    days, hours = r.randrange(2, 9), r.randrange(2, 13)
    q = f"How many minutes are there in {days} days and {hours} hours?"
    return q, f"{days} * 24 * 60 + {hours} * 60"


def _hard() -> tuple[str, str]:
    r = _RNG
    kind = r.randrange(9)
    if kind == 0:
        a, b = r.randrange(10000, 99999), r.randrange(100, 999)
        return f"What is {a} * {b}?", f"{a} * {b}"
    if kind == 1:
        a, b = r.randrange(1000, 9999), r.randrange(1000, 9999)
        return f"What is {a} * {b}?", f"{a} * {b}"
    if kind == 2:
        a, b = r.randrange(11, 40), r.randrange(4, 7)
        return f"What is {a} to the power of {b}?", f"{a} ** {b}"
    if kind == 3:
        a, c = (r.randrange(5, 60) + r.choice((0.25, 0.5, 0.75)) for _ in range(2))
        e, g, f = r.randrange(5, 30), r.randrange(3, 20), r.randrange(100, 900)
        return f"What is ({a} * {e}) - ({c} * {g}) + {f}?", f"({a} * {e}) - ({c} * {g}) + {f}"
    if kind == 4:
        a, b = r.randrange(500, 9999), r.randrange(7, 97)
        return f"What is {a} divided by {b}, rounded to 2 decimal places?", f"{a} / {b}"
    if kind == 5:
        p, rate, yrs = r.randrange(500, 5000, 50), r.randrange(2, 9), r.randrange(3, 12)
        q = (
            f"{p} is invested at {rate}% interest compounded yearly for {yrs} years. "
            "What is the final amount, rounded to 2 decimal places?"
        )
        return q, f"{p} * {(100 + rate) / 100} ** {yrs}"
    if kind == 6:
        n = r.randrange(100, 900)
        return f"What is the sum of all whole numbers from 1 to {n}?", f"{n} * {n + 1} / 2"
    if kind == 7:
        a, b = r.randrange(10000, 999999), r.randrange(31, 199)
        return f"What is the remainder when {a} is divided by {b}?", f"{a} % {b}"
    n = r.randrange(6, 13)
    squares = " + ".join(f"{k} ** 2" for k in range(1, n + 1))
    return f"What is the sum of the squares of the whole numbers from 1 to {n}?", squares


def _fact_math() -> tuple[str, str]:
    r = _RNG
    kind = r.randrange(10)
    if kind == 0:
        y = r.randrange(2026, 2060)
        return f"How many years pass from the company's founding to {y}?", f"{y} - 2024"
    if kind == 1:
        n = r.randrange(3, 60)
        q = f"The company hires {n} more people. How many employees does it then have?"
        return q, f"42 + {n}"
    if kind == 2:
        rev = 42 * r.randrange(100, 900)
        return f"Revenue is {rev}. What is the revenue per employee?", f"{rev} / 42"
    if kind == 3:
        m = r.randrange(2, 6)
        q = f"The company multiplies its headcount by {m}. How many employees is that?"
        return q, f"42 * {m}"
    if kind == 4:
        n = r.randrange(2, 40)
        return f"What do {n} units of the company's product cost?", f"{n} * 19.99"
    if kind == 5:
        n, off = r.randrange(5, 50), r.choice((5, 10, 15, 20, 25))
        q = f"What do {n} units of the product cost after a {off}% discount?"
        return q, f"{n} * 19.99 * {(100 - off) / 100}"
    if kind == 6:
        budget = r.randrange(50, 900)
        q = f"How many whole units of the product can you buy with {budget}?"
        return q, f"{budget} // 19.99"
    if kind == 7:
        n, k, c = r.randrange(2, 9), r.randrange(2, 9), r.randrange(100, 900)
        q = (
            f"A customer buys {n} units of the product and {k} gadgets at {_money(c)} each. "
            "What is the total?"
        )
        return q, f"{n} * 19.99 + {k} * {_money(c)}"
    if kind == 8:
        wage = r.randrange(30, 90) * 1000
        q = f"If each employee costs {wage} a year, what is the yearly staff cost?"
        return q, f"42 * {wage}"
    n = r.randrange(2, 20)
    q = f"If each employee sells {n} units of the product, how many units is that in total?"
    return q, f"42 * {n}"


_fill(EASY, 30, _easy)
_fill(MEDIUM, 45, _medium)
_fill(HARD, 35, _hard)
_fill(FACT_MATH, 35, _fact_math)

FACT += [
    ("Name the city where the company has its headquarters.", "headquarters"),
    ("Which city houses the company's head office?", "headquarters"),
    ("Where is the company based?", "headquarters"),
    ("What year did the company start?", "founded"),
    ("In what year was the company established?", "founded"),
    ("The company was founded in which year?", "founded"),
    ("How large is the company's workforce?", "employees"),
    ("How many people work at the company?", "employees"),
    ("What is the company's employee count?", "employees"),
    ("What does the company make?", "product"),
    ("Name the company's main product.", "product"),
    ("What is the price of one unit of the product, in USD?", "widget_price_usd"),
    ("How many US dollars does a single unit of the product cost?", "widget_price_usd"),
    ("What does the company charge per unit, in USD?", "widget_price_usd"),
]

_PRIMES = (
    101, 109, 127, 139, 151, 163, 179, 191, 197, 211, 227, 233, 241, 257, 271, 283,
)  # fmt: skip
_COMPOSITES = (
    111, 119, 133, 147, 161, 187, 203, 217, 247, 259, 287, 299, 319, 341, 377, 403,
)  # fmt: skip
YES_NO += [(f"Is {p} a prime number?", "yes") for p in _PRIMES[:8]]
YES_NO += [(f"Is {c} a prime number?", "no") for c in _COMPOSITES[:8]]
YES_NO += [(f"Is the company headquartered in {c}?", "no") for c in ("Madrid", "Paris", "Oslo")]
YES_NO += [(f"Does the company have more than {n} employees?", "yes") for n in (30, 41)]
YES_NO += [(f"Does the company have more than {n} employees?", "no") for n in (42, 50, 100)]
YES_NO += [(f"Was the company founded before {y}?", "yes") for y in (2025, 2030)]
YES_NO += [(f"Was the company founded before {y}?", "no") for y in (2024, 2010)]
YES_NO += [(f"Does one unit of the product cost less than {p} USD?", "yes") for p in (25, 30)]
YES_NO += [(f"Does one unit of the product cost less than {p} USD?", "no") for p in (10, 19)]

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


MARKER = re.compile(r"\\b(?:final\\s+)?answer\\s*(?:is\\s*)?[:=]", re.IGNORECASE)


def extract(text):
    # After an explicit "Answer:" marker the first number is the answer; otherwise the last
    # number is (a sentence that ends with the result, as the question asks).
    marker = MARKER.search(text)
    after = NUMBER.findall(text[marker.end() :]) if marker else []
    found = [after[0]] if after else NUMBER.findall(text)[-1:]
    if not found:
        raise ValueError("no number")
    d = Decimal(found[0].replace(",", "")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    out = f"{d:f}"
    out = out.rstrip("0").rstrip(".") if "." in out else out
    return {"number": out}
"""

YESNO_BODY = """
VERDICT = re.compile(r"\\b(yes|no)\\b", re.IGNORECASE)


def extract(text):
    found = VERDICT.search(text)  # the first yes/no is the verdict; later words are explanation
    if not found:
        raise ValueError("no verdict")
    return {"verdict": found.group(1).lower()}
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
    for stale in ("tasks", "pack.toml", "README.md", "splits.json"):
        path = root / stale
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    (root / "tasks").mkdir(parents=True)
    me = root / "generate.py"
    if not me.exists() or not me.samefile(Path(__file__)):
        shutil.copyfile(Path(__file__), me)
    (root / "pack.toml").write_bytes(
        f'name = "reference-qa"\nversion = "1"\ngrader_image = "{GRADER_IMAGE}"\n'.encode()
    )
    (root / "README.md").write_bytes(
        b"# reference-qa\n\n"
        b"Small question-answering pack for the reference agent (arithmetic, the demo\n"
        b"company's facts, yes/no). Written for Iterant and licensed Apache-2.0 like the rest\n"
        b"of the repo. Regenerate with `generate.py` in this directory.\n\n"
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
