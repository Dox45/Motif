from .types import Buffer
from .frontend import kernel, Kernel, MotifSyntaxError


def global_id(axis: int):
    """Marker function recognized syntactically by the compiler in
    `x = motif.global_id(axis)`. Never actually called at runtime — the
    @kernel decorator compiles the function's *source*, it does not
    execute the body. This stub exists only so plain `python examples/foo.py`
    doesn't NameError if someone runs a kernel file directly, and so your
    editor's autocomplete/type-checker sees a real symbol.
    """
    raise RuntimeError(
        "motif.global_id() has no runtime implementation — it is compiled "
        "away by @motif.kernel. If you're seeing this, the function was "
        "called directly instead of through the kernel decorator."
    )


__all__ = ["kernel", "Kernel", "Buffer", "global_id", "MotifSyntaxError"]
