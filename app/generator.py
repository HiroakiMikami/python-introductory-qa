import string
import numpy as np
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class Function:
    name: str
    obj: Optional[str]
    args: List[List[str]]
    return_type: List[str]


@dataclass
class Operator:
    name: str
    name_str: str
    args: List[List[str]]
    return_type: List[str]


# Types
any = ["str", "int", "float", "bool", "list"]
number = ["int", "float"]

# Variables
variables = {
    "func": ["f", "g"],
    "any": ["x", "y", "z"],
    "number": ["x", "y", "z"],
    "str": ["x", "y", "s", "t"],
    "int": ["x", "y", "i", "n"],
    "float": ["x", "y", "z", "f"],
    "bool": ["x", "y", "c", "cond"],
    "list": ["xs", "ys", "values"],
}

# Builtin Functions
functions = [
    Function("print", None, [any], []),
    Function("append", "list", [any], []),
    Function("insert", "list", [
             ["int"], any], []),
    Function("len", None, [["list"]], ["int"]),
    Function("str", None, [any], ["str"]),
    Function("range", None, [["int"]], ["list"]),
    Function("@get", None, [["list"], ["int"]], any),
    Function("@del", None, [["list"], ["int"]], []),
]
operators = [
    # Operator("+", "plus", [["int"], ["int"]], "int"),
    # Operator("-", "minus", [["int"], ["int"]], "int"),
    # Operator("*", "multiply", [["int"], ["int"]], "int"),
    Operator("+", "plus", [number, number], number),
    Operator("-", "minus", [number, number], number),
    Operator("*", "multiply", [number, number], number),
    Operator("/", "divide", [number, number], number),
    Operator("//", "divide as integer", [number, number], ["int"]),
    Operator("%", "modulo", [["int"], ["int"]], ["int"]),
    Operator("**", "power of", [number, number], number),
    Operator("in", "in", [any, ["list"]], ["bool"]),
    Operator("is", "is", [any, any], ["bool"]),
    Operator("==", "equal", [any, any], ["bool"]),
    Operator("!=", "not equal", [any, any], ["bool"]),
]
inplaces = ["+", "-", "*", "/", "//", "%"]


@dataclass
class Example:
    text: str
    code: str


class Generator:
    def __init__(self, rng: np.random.RandomState):
        self.rng = rng

    def _create_variable(self, t: str) -> Example:
        v = self.rng.choice(variables[t])
        return Example(v, v)

    def _create_constant(self, t: str) -> Example:
        if t == "int":
            i = str(self.rng.randint(0, 100))
            return Example(i, i)
        elif t == "float":
            f = "{:.2f}".format(self.rng.rand())
            return Example(f, f)
        elif t == "str":
            n = self.rng.randint(0, 10)
            t = ''.join(self.rng.choice(list(string.ascii_letters), size=n))
            return Example(t, f'"{t}"')
        elif t == "bool":
            b = self.rng.choice(["True", "False"])
            return Example(b.lower(), b)
        elif t == "list":
            n = self.rng.randint(0, 3)
            if n == 0:
                return Example("the empty list", "[]")
            else:
                subtype = self.rng.choice(any)
                text = "the list of"
                code = "["
                for elem in [self._create_constant(subtype) for _ in range(n)]:
                    text += f" {elem.text}"
                    code += f"{elem.code}, "
                code += "]"
                return Example(text, code)
        raise AssertionError(f"invalid type: {t}")

    def _create_function_call(self, f: Optional[Function] = None) -> Example:
        if f is None:
            f = self.rng.choice(functions)

        if f.name.startswith("@"):
            # specific function
            args = [self._create_constant(self.rng.choice(ts))
                    for ts in f.args]
            if f.name == "@get":
                return Example(
                    f"{args[1].text} element of {args[0].text}",
                    f"{args[0].code}[{args[1].code}]"
                )
            elif f.name == "@del":
                return Example(
                    f"delete {args[1].text} element from {args[0].text}",
                    f"del {args[0].code}[{args[1].code}]"
                )
            raise AssertionError(f"invalid function name: {f.name}")
        if f.obj is None:
            # function
            text = f"{f.name}"
            code = f"{f.name}("
        else:
            # method
            obj = self._create_variable(f.obj)
            text = f"{obj.text} {f.name}"
            code = f"{obj.text}.{f.name}("
        for arg in [self._create_constant(self.rng.choice(ts))
                    for ts in f.args]:
            text += f" {arg.text}"
            code += f"{arg.code},"
        code += ")"
        return Example(text, code)

    def _create_operator(self, op: Optional[Operator] = None) -> Example:
        if op is None:
            op = self.rng.choice(operators)
        use_str = self.rng.choice([False, True])
        if use_str:
            op_str = op.name_str
        else:
            op_str = op.name

        text = ""
        code = "("
        args = [self._create_constant(self.rng.choice(ts)) for ts in op.args]
        for i, arg in enumerate(args):
            text += arg.text
            code += arg.code
            if i != len(args) - 1:
                text += f" {op_str} "
                code += f" {op.name} "
        code += ")"
        return Example(text, code)

    def _create_expression(self, types: List[str]) -> Example:
        t = self.rng.choice(types)
        funcs = [f for f in functions
                 if t in f.return_type or f.return_type == types]
        ops = [f for f in operators
               if t in f.return_type or f.return_type == types]
        cands = ["constant"] + ["variable"]
        if len(funcs) != 0:
            cands.append("function")
        if len(ops) != 0:
            cands.append("operator")
        c = self.rng.choice(cands)

        if c == "constant":
            return self._create_constant(t)
        elif c == "variable":
            return self._create_variable(t)
        elif c == "function":
            return self._create_function_call(self.rng.choice(funcs))
        elif c == "operator":
            return self._create_operator(self.rng.choice(ops))
        raise AssertionError(f"invalid candidate: {c}")

    def _create_assign(self) -> Example:
        t = self.rng.choice(any + ["any", "number"])
        x = self._create_variable(t)
        if t == "number":
            ts = number
        elif t == "any":
            ts = any
        else:
            ts = [t]
        v = self._create_expression(ts)
        return Example(
            f"assign {x.text} with {v.text}",
            f"{x.code} = {v.code}"
        )

    def _create_inplace(self) -> Example:
        inplace_op = self.rng.choice(inplaces)
        op = [op for op in operators if op.name == inplace_op][0]
        t = self.rng.choice(op.args[0])
        x = self._create_variable(t)
        v = self._create_expression(op.args[1])
        return Example(
            f"assign {x.text} with {x.text} {op.name} {v.text}",
            f"{x.code} {op.name}= {v.code}"
        )

    def _create_suite(self) -> Example:
        n = self.rng.randint(1, 3)
        funcs = [f for f in functions if f.return_type == []]
        funcs.extend(["assign", "inplace"])
        text = ""
        code = ""
        for i in range(n):
            x = self.rng.choice(funcs)
            if isinstance(x, Function):
                e = self._create_function_call(x)
            elif x == "assign":
                e = self._create_assign()
            elif x == "inplace":
                e = self._create_inplace()
            text += e.text
            code += e.code
            if i != (n - 1):
                text += ", "
                code += "\n"
        return Example(text, code)

    def _indent(self, text: str) -> str:
        lines = text.split("\n")
        lines = ["  " + line for line in lines]
        return "\n".join(lines)

    def _create_if(self) -> Example:
        has_elif = self.rng.choice([False, True])
        has_else = self.rng.choice([False, True])

        text = ""
        code = ""

        # if statement
        c = self._create_expression(["bool"])
        body = self._create_suite()
        if len(body.code.split("\n")) == 1:
            text += f"{body.text} if {c.text}."
        else:
            text += f"if {c.text} do followings: {body.text}."
        code += f"if {c.code}:\n" + self._indent(body.code)

        # elif statement
        if has_elif:
            c = self._create_expression(["bool"])
            body = self._create_suite()
            if len(body.code.split("\n")) == 1:
                text += f"{body.text} if {c.text}."
            else:
                text += f"if {c.text} do followings: {body.text}."
            code += f"\nelif {c.code}:\n" + self._indent(body.code)

        # else statement
        if has_else:
            body = self._create_suite()
            if len(body.code.split("\n")) == 1:
                text += f"otherwise {body.text}."
            else:
                text += f"otherwise do followings: {body.text}."
            code += "\nelse:\n" + self._indent(body.code)
        return Example(text, code)

    def _create_while(self):
        c = self._create_expression(["bool"])
        body = self._create_suite()
        text = ""
        code = ""
        if len(body.code.split("\n")) == 1:
            text += f"{body.text} while {c.text}."
        else:
            text += f"while {c.text} do followings: {body.text}."
        code += f"while {c.code}:\n" + self._indent(body.code)
        return Example(text, code)

    def _create_for(self):
        x = self._create_variable("any")
        xs = self._create_expression(["list"])
        body = self._create_suite()
        text = ""
        code = ""
        if len(body.code.split("\n")) == 1:
            text += f"{body.text} for each {x.text} in {xs.text}."
        else:
            text += \
                f"for each {x.text} in {xs.text} do followings: {body.text}."
        code += f"for {x.code} in {xs.code}:\n" + self._indent(body.code)
        return Example(text, code)

    def _create_funcdef(self):
        name = self._create_variable("func")
        n_arg = self.rng.randint(0, 3)
        args = [self._create_variable("any") for _ in range(n_arg)]

        body = self._create_suite()
        text = ""
        code = ""
        if len(body.code.split("\n")) == 1:
            text += f"{body.text} when {name.text}."
        else:
            text += \
                f"when {name.text} is called do followings: {body.text}."
        args_code = ",".join([arg.code for arg in args])
        code += f"def {name.code}({args_code}):\n" + self._indent(body.code)
        return Example(text, code)

    def create(self) -> Example:
        cands = [self._create_funcdef, self._create_for, self._create_while,
                 self._create_if, self._create_assign, self._create_inplace,
                 self._create_function_call, self._create_operator]
        c = self.rng.choice(cands)
        return c()
