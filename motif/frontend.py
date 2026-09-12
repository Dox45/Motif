"""
Motif v0.1 frontend.

Pipeline:  Python source --(inspect)--> ast.FunctionDef
                        --(CodeGen visitor)--> GLSL compute shader text

This is deliberately a *transliterator*, not a general compiler. It only
understands the handful of node types needed to express vecadd/matmul-shaped
kernels (see shaders/vecadd.comp and shaders/matmul.comp for the ground
truth this must produce). Anything outside that whitelist raises
MotifSyntaxError loudly rather than guessing — an unsupported construct
should fail at compile time, not produce silently wrong GLSL.

Extend this file by extending the whitelist, one construct at a time, only
when a real kernel needs it. Don't pre-build support for constructs no
example uses yet.
"""

import ast
import inspect
import textwrap

from .types import Buffer, GLSL_SCALAR_TYPES

_AXIS_SWIZZLE = ("x", "y", "z")

_BINOP = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
    ast.Mod: "%",
}
_CMPOP = {
    ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
    ast.Eq: "==", ast.NotEq: "!=",
}

# GLSL reserves a handful of short, very Python-idiomatic names as storage
# qualifiers/keywords. `out` in particular is exactly the name anyone will
# reach for on their first kernel (see hand-written vecadd.comp, which had
# to use `out_`). Rather than make every kernel author discover this the
# hard way, the compiler auto-suffixes a collision — the user still writes
# and reads `out` in Python; only the emitted GLSL identifier changes.
_GLSL_RESERVED = {"in", "out", "input", "output", "buffer", "uniform", "const"}


def _glsl_safe(name: str) -> str:
    return f"{name}_" if name in _GLSL_RESERVED else name


class MotifSyntaxError(Exception):
    """Raised when a kernel uses a Python construct outside the v0.1
    supported subset. The message always names the offending node type
    so the fix is obvious."""


class Kernel:
    """Result of compiling one @motif.kernel function.

    glsl        - the generated compute shader source (str)
    dispatch_dims - number of axes read via motif.global_id(axis); tells
                    the caller whether to dispatch 1D or 2D
    local_size  - workgroup size tuple, matching dispatch_dims
    buffers     - ordered list of buffer param names (binding index ==
                  position in this list)
    scalars     - ordered list of (name, py_type) for push-constant params
    """

    def __init__(self, name, glsl, dispatch_dims, local_size, buffers, scalars):
        self.name = name
        self.glsl = glsl
        self.dispatch_dims = dispatch_dims
        self.local_size = local_size
        self.buffers = buffers
        self.scalars = scalars

    def __repr__(self):
        return f"<Kernel {self.name} dims={self.dispatch_dims} " \
               f"buffers={self.buffers} scalars={self.scalars}>"


def kernel(fn):
    """Decorator: compiles fn's Python source to a GLSL Kernel at import
    time (eagerly — no lazy tracing in v0.1, so errors surface immediately
    next to the kernel definition, not at first dispatch)."""
    source = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(source)
    func_def = tree.body[0]
    if not isinstance(func_def, ast.FunctionDef):
        raise MotifSyntaxError("@motif.kernel must decorate a function")

    compiled = _CodeGen(func_def).compile()
    compiled.__wrapped_source__ = source  # kept for debugging / printing
    return compiled


class _CodeGen(ast.NodeVisitor):
    def __init__(self, func_def: ast.FunctionDef):
        self.func_def = func_def
        self.buffers = []          # [name, ...]  binding index = list index
        self.scalars = []          # [(name, py_type), ...]
        self.locals = {}           # name -> glsl type ("uint" | "float")
        self.global_id_vars = {}   # name -> axis int (assigned from motif.global_id(k))
        self.dispatch_dims = 0
        self.lines = []
        self.indent = 1

    # ---- entry point ---------------------------------------------------

    def compile(self) -> Kernel:
        self._parse_signature(self.func_def)
        self._scan_dispatch_dims(self.func_def.body)
        local_size = self._pick_local_size()

        header = [
            "#version 450",
            f"layout({', '.join(f'local_size_{a}={n}' for a, n in zip(_AXIS_SWIZZLE, local_size))}) in;",
            "",
        ]
        for i, name in enumerate(self.buffers):
            glsl_name = _glsl_safe(name)
            header.append(f"layout(binding = {i}) buffer Buf{glsl_name.capitalize()} {{ float {glsl_name}[]; }};")
        header.append("")

        if self.scalars:
            header.append("layout(push_constant) uniform Params {")
            for name, py_type in self.scalars:
                header.append(f"    {GLSL_SCALAR_TYPES[py_type]} {name};")
            header.append("} params;")
            header.append("")

        header.append("void main() {")
        for name, axis in self.global_id_vars.items():
            header.append(f"    uint {name} = gl_GlobalInvocationID.{_AXIS_SWIZZLE[axis]};")

        for stmt in self.func_def.body:
            self.visit(stmt)

        body = header + self.lines + ["}"]
        glsl = "\n".join(body) + "\n"

        return Kernel(
            name=self.func_def.name,
            glsl=glsl,
            dispatch_dims=self.dispatch_dims,
            local_size=local_size,
            buffers=list(self.buffers),
            scalars=list(self.scalars),
        )

    # ---- signature / pre-scan ------------------------------------------

    def _parse_signature(self, func_def):
        for arg in func_def.args.args:
            ann = arg.annotation
            if ann is None:
                raise MotifSyntaxError(
                    f"param '{arg.arg}' needs a type annotation (Buffer, int, or float)")
            if isinstance(ann, ast.Name) and ann.id == "Buffer":
                self.buffers.append(arg.arg)
            elif isinstance(ann, ast.Name) and ann.id in ("int", "float"):
                py_type = int if ann.id == "int" else float
                self.scalars.append((arg.arg, py_type))
            else:
                raise MotifSyntaxError(
                    f"param '{arg.arg}': unsupported annotation "
                    f"'{ast.dump(ann)}' (v0.1 supports Buffer, int, float)")

    def _scan_dispatch_dims(self, body):
        """First pass: find every `<name> = motif.global_id(<axis>)` (or
        bare `global_id(<axis>)`) so we know dispatch rank and can resolve
        those names to gl_GlobalInvocationID.* before the real visit."""
        for node in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                axis = self._match_global_id_call(node.value)
                if axis is not None and len(node.targets) == 1 \
                        and isinstance(node.targets[0], ast.Name):
                    name = node.targets[0].id
                    self.global_id_vars[name] = axis
                    self.locals[name] = "uint"
                    self.dispatch_dims = max(self.dispatch_dims, axis + 1)

    @staticmethod
    def _match_global_id_call(call: ast.Call):
        fn = call.func
        is_global_id = (
            (isinstance(fn, ast.Attribute) and fn.attr == "global_id") or
            (isinstance(fn, ast.Name) and fn.id == "global_id")
        )
        if not is_global_id or len(call.args) != 1:
            return None
        arg = call.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
            return arg.value
        raise MotifSyntaxError("global_id(axis) requires a literal int axis")

    def _pick_local_size(self):
        # v0.1 heuristic: 256 threads for 1D, 16x16 for 2D. Not tuned —
        # this is a starting point, not Triton's auto-tuner (Sec 5.3).
        if self.dispatch_dims <= 1:
            return (256,)
        elif self.dispatch_dims == 2:
            return (16, 16)
        raise MotifSyntaxError("v0.1 supports at most 2D dispatch")

    # ---- statements ------------------------------------------------------

    def _emit(self, line):
        self.lines.append("    " * self.indent + line)

    def visit_Assign(self, node: ast.Assign):
        if len(node.targets) != 1:
            raise MotifSyntaxError("multiple assignment targets not supported")
        target = node.targets[0]

        if isinstance(target, ast.Name):
            if target.id in self.global_id_vars:
                return  # already emitted in the header during compile()
            expr, ty = self._expr(node.value)
            if target.id in self.locals:
                self._emit(f"{target.id} = {expr};")
            else:
                self.locals[target.id] = ty
                self._emit(f"{ty} {target.id} = {expr};")

        elif isinstance(target, ast.Subscript):
            buf_expr = self._expr(target.value)[0]
            index_expr = self._expr(target.slice)[0]
            value_expr = self._expr(node.value)[0]
            self._emit(f"{buf_expr}[{index_expr}] = {value_expr};")

        else:
            raise MotifSyntaxError(f"unsupported assignment target: {ast.dump(target)}")

    def visit_AugAssign(self, node: ast.AugAssign):
        if not isinstance(node.target, ast.Name):
            raise MotifSyntaxError("augmented assignment only supported on local scalars")
        op = _BINOP.get(type(node.op))
        if op is None:
            raise MotifSyntaxError(f"unsupported augmented op: {ast.dump(node.op)}")
        expr, _ = self._expr(node.value)
        self._emit(f"{node.target.id} {op}= {expr};")

    def visit_If(self, node: ast.If):
        if node.orelse:
            raise MotifSyntaxError("else branches not supported in v0.1")
        cond, _ = self._expr(node.test)
        self._emit(f"if ({cond}) {{")
        self.indent += 1
        for stmt in node.body:
            self.visit(stmt)
        self.indent -= 1
        self._emit("}")

    def visit_For(self, node: ast.For):
        if not isinstance(node.target, ast.Name):
            raise MotifSyntaxError("for-loop target must be a plain name")
        if not (isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range" and len(node.iter.args) == 1):
            raise MotifSyntaxError("v0.1 only supports `for x in range(N):`")
        var = node.target.id
        bound, _ = self._expr(node.iter.args[0])
        self.locals[var] = "uint"
        self._emit(f"for (uint {var} = 0; {var} < {bound}; {var}++) {{")
        self.indent += 1
        for stmt in node.body:
            self.visit(stmt)
        self.indent -= 1
        self._emit("}")

    def generic_visit(self, node):
        raise MotifSyntaxError(f"unsupported statement: {type(node).__name__}")

    # ---- expressions -------------------------------------------------
    # Each returns (glsl_string, glsl_type)

    def _expr(self, node):
        if isinstance(node, ast.BoolOp):
            op = "&&" if isinstance(node.op, ast.And) else "||"
            parts = [self._expr(v)[0] for v in node.values]
            return "(" + f" {op} ".join(parts) + ")", "bool"

        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise MotifSyntaxError("chained comparisons not supported")
            op = _CMPOP.get(type(node.ops[0]))
            if op is None:
                raise MotifSyntaxError(f"unsupported comparison: {ast.dump(node.ops[0])}")
            left, _ = self._expr(node.left)
            right, _ = self._expr(node.comparators[0])
            return f"{left} {op} {right}", "bool"

        if isinstance(node, ast.BinOp):
            op = _BINOP.get(type(node.op))
            if op is None:
                raise MotifSyntaxError(f"unsupported operator: {ast.dump(node.op)}")
            left, lty = self._expr(node.left)
            right, rty = self._expr(node.right)
            ty = "float" if "float" in (lty, rty) else "uint"
            return f"({left} {op} {right})", ty

        if isinstance(node, ast.Subscript):
            buf, _ = self._expr(node.value)
            idx, _ = self._expr(node.slice)
            return f"{buf}[{idx}]", "float"

        if isinstance(node, ast.Name):
            if node.id in self.global_id_vars or node.id in self.locals:
                return node.id, self.locals.get(node.id, "uint")
            if node.id in self.buffers:
                return _glsl_safe(node.id), "buffer"
            for name, py_type in self.scalars:
                if node.id == name:
                    return f"params.{name}", GLSL_SCALAR_TYPES[py_type]
            raise MotifSyntaxError(f"unknown identifier '{node.id}'")

        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return ("true" if node.value else "false"), "bool"
            if isinstance(node.value, float):
                return repr(node.value), "float"
            if isinstance(node.value, int):
                return str(node.value), "uint"
            raise MotifSyntaxError(f"unsupported constant: {node.value!r}")

        raise MotifSyntaxError(f"unsupported expression: {type(node).__name__}")
