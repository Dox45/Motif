"""
Motif v0.1 type markers.

These carry no runtime behavior — they exist purely so the AST-based
compiler can read a kernel's Python type annotations and decide:
  - which params become GLSL storage buffers (Buffer)
  - which params become push-constant scalars (int / float)

This mirrors how Triton-C's `float* a` vs `int M` in Listing 1 tells the
compiler "this is a tile of pointers" vs "this is a scalar" — we're just
doing it via Python annotations instead of C syntax.
"""


class Buffer:
    """Marks a kernel parameter as a GPU storage buffer (SSBO).

    Usage:
        def vecadd(a: Buffer, b: Buffer, out: Buffer, n: int): ...

    v0.1 assumes float32 elements for every buffer. Per-buffer dtype
    (int32, float16, ...) is a v0.2 concern — don't add it until a
    kernel actually needs it.
    """
    dtype = "float"


# Scalar params use plain Python types in the annotation: `n: int`,
# `alpha: float`. The compiler maps int -> GLSL `uint`, float -> GLSL
# `float`. Signed int and other widths are not v0.1 scope.
GLSL_SCALAR_TYPES = {
    int: "uint",
    float: "float",
}
