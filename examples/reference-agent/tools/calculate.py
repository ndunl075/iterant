"""Safe arithmetic: parses the expression and allows only numbers and + - * / % **."""

import ast
import operator

SCHEMA = {
    "name": "calculate",
    "description": "Evaluate an arithmetic expression such as '19.99 * 3 + 2'.",
    "input_schema": {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    },
}

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("only numbers and + - * / % ** are allowed")


def calculate(expression: str) -> float:
    return _eval(ast.parse(expression, mode="eval").body)
