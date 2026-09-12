"""
Sanity check for the v0.1 frontend — no GPU, no glslang required.

This does NOT prove the GLSL is valid (you still need glslangValidator /
glslc, or Vulkan's own shader module creation, to confirm that — run this
on your dev machine or in the Android build once you get here). What it
proves is that the compiler's output is *structurally* what vecadd/matmul
need: right bindings, right push constants, right invocation ID usage,
right loop. Run it after every change to frontend.py so regressions show
up immediately instead of at the next on-device dispatch.
"""

import sys
sys.path.insert(0, ".")

from examples.vecadd import vecadd
from examples.matmul import matmul


from motif.dispatch import generate_cpp_dispatch


def check(condition, msg):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {msg}")
    return condition


def test_vecadd():
    print("\n-- vecadd --")
    ok = True
    ok &= check(vecadd.dispatch_dims == 1, "1D dispatch")
    ok &= check(vecadd.local_size == (256,), "local_size_x = 256")
    ok &= check(vecadd.buffers == ["a", "b", "out"], "buffer order/bindings a=0,b=1,out=2")
    ok &= check(vecadd.scalars == [("n", int)], "scalar push-constant: n")
    ok &= check("gl_GlobalInvocationID.x" in vecadd.glsl, "uses global invocation id")
    ok &= check("out_[i] = (a[i] + b[i]);" in vecadd.glsl, "correct elementwise add + reserved-word escape")
    ok &= check("if (i < params.n)" in vecadd.glsl, "bounds check against push constant")

    cpp_code = generate_cpp_dispatch(vecadd)
    ok &= check("void run_vecadd_kernel" in cpp_code, "C++ dispatch function signature generated")
    ok &= check("kp::Manager mgr;" in cpp_code, "Kompute kp::Manager instantiated in C++")
    ok &= check("OpAlgoDispatch" in cpp_code, "Kompute OpAlgoDispatch present in C++")
    return ok


def test_matmul():
    print("\n-- matmul --")
    ok = True
    ok &= check(matmul.dispatch_dims == 2, "2D dispatch")
    ok &= check(matmul.local_size == (16, 16), "local_size 16x16")
    ok &= check(matmul.buffers == ["a", "b", "c"], "buffer order/bindings a=0,b=1,c=2")
    ok &= check([s[0] for s in matmul.scalars] == ["M", "N", "K"], "scalar order M,N,K")
    ok &= check("gl_GlobalInvocationID.y" in matmul.glsl, "2D dispatch uses .y for second axis")
    ok &= check("for (uint k = 0; k < params.K; k++)" in matmul.glsl, "reduction loop over K")
    ok &= check("acc += " in matmul.glsl, "accumulator uses +=")

    cpp_code = generate_cpp_dispatch(matmul)
    ok &= check("void run_matmul_kernel" in cpp_code, "C++ matmul dispatch function signature generated")
    ok &= check("matmul_push_constants" in cpp_code, "C++ matmul push constants struct generated")
    return ok


if __name__ == "__main__":
    results = [test_vecadd(), test_matmul()]
    print("\n" + ("ALL PASS" if all(results) else "SOME FAILED"))
    sys.exit(0 if all(results) else 1)

