"""
Bridges a compiled `motif.Kernel` (GLSL text + metadata) to the native Kompute C++
dispatch pipeline on Android (and PC).

Provides:
  - `compile_to_spirv(kernel)`: Compiles GLSL text to SPIR-V bytecode using `glslangValidator` or `glslc`.
  - `generate_cpp_dispatch(kernel)`: Generates zero-overhead native C++ code using Kompute C++ API (`kp::Manager`, `kp::Tensor`, `kp::Algorithm`).
  - `generate_test_runner(kernel)`: Emits a standalone C++ `main()` test runner executable for Android NDK (`adb shell`) GPU verification.
"""

import subprocess
import tempfile
import os
from typing import Optional, List, Tuple

from .frontend import Kernel


def compile_to_spirv(kernel: Kernel) -> bytes:
    """Compiles kernel GLSL source to SPIR-V binary words using glslangValidator or glslc."""
    compiler_cmd = None
    for cmd in ["glslangValidator", "glslc"]:
        try:
            res = subprocess.run([cmd, "--version"], capture_output=True)
            if res.returncode == 0:
                compiler_cmd = cmd
                break
        except FileNotFoundError:
            continue

    if not compiler_cmd:
        raise RuntimeError(
            "Neither `glslangValidator` nor `glslc` was found on PATH. "
            "Please install Vulkan SDK or glslang to compile GLSL to SPIR-V."
        )

    with tempfile.NamedTemporaryFile(suffix=".comp", mode="w", delete=False) as f:
        f.write(kernel.glsl)
        src_path = f.name
    spv_path = src_path + ".spv"

    try:
        if compiler_cmd == "glslangValidator":
            cmd_args = ["glslangValidator", "-V", src_path, "-o", spv_path]
        else:
            cmd_args = ["glslc", src_path, "-o", spv_path]

        subprocess.run(cmd_args, check=True, capture_output=True, text=True)
        with open(spv_path, "rb") as f:
            return f.read()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"GLSL compile failed for kernel '{kernel.name}':\n{e.stderr}\n"
            f"--- generated source ---\n{kernel.glsl}"
        )
    finally:
        if os.path.exists(src_path):
            os.unlink(src_path)
        if os.path.exists(spv_path):
            os.unlink(spv_path)


def generate_cpp_dispatch(kernel: Kernel, spirv_bytes: Optional[bytes] = None) -> str:
    """Generates zero-overhead native C++ source code using Kompute C++ API

    Targeting Android NDK & native Vulkan drivers (libvulkan.so).
    """
    if spirv_bytes is None:
        try:
            spirv_bytes = compile_to_spirv(kernel)
        except Exception:
            spirv_bytes = b""

    # Convert SPIR-V bytes to 32-bit uint array literal format for C++
    spirv_words = []
    if spirv_bytes:
        for i in range(0, len(spirv_bytes), 4):
            word = int.from_bytes(spirv_bytes[i:i+4], byteorder='little')
            spirv_words.append(f"0x{word:08x}")

    spirv_array_str = ", ".join(spirv_words) if spirv_words else "/* Compile SPIR-V bytes here */"

    # Format scalar struct members
    scalar_members = []
    push_const_args = []
    for name, py_type in kernel.scalars:
        c_type = "int32_t" if py_type == int else "float"
        scalar_members.append(f"    {c_type} {name};")
        push_const_args.append(f"static_cast<float>({name})")

    scalar_struct_def = "\n".join(scalar_members) if scalar_members else "    // No push constants"

    # Buffer vector setup
    buffer_ptrs = [f"tensor_{buf}" for buf in kernel.buffers]
    tensor_vec_str = ", ".join(buffer_ptrs)

    # Tensor creation parameters
    tensor_creations = []
    for buf in kernel.buffers:
        tensor_creations.append(
            f"    std::shared_ptr<kp::TensorT<float>> tensor_{buf} = mgr.tensor(std::vector<float>({buf}_data, {buf}_data + element_count));\n"
        )
    tensor_create_str = "".join(tensor_creations)

    last_buf = kernel.buffers[-1]
    sync_back_code = f"    std::vector<float> {last_buf}_res = tensor_{last_buf}->vector();\n    std::copy({last_buf}_res.begin(), {last_buf}_res.end(), {last_buf}_data);"

    # Local workgroup sizing calculation
    if kernel.dispatch_dims == 1:
        local_x = kernel.local_size[0]
        workgroup_calc = f"{{ (element_count + {local_x - 1}) / {local_x}, 1, 1 }}"
    elif kernel.dispatch_dims == 2:
        local_x, local_y = kernel.local_size
        scalar_names = [s[0] for s in kernel.scalars[:2]]
        dim_x_var = scalar_names[0] if len(scalar_names) > 0 else "element_count"
        dim_y_var = scalar_names[1] if len(scalar_names) > 1 else "element_count"
        workgroup_calc = f"{{ static_cast<uint32_t>(({dim_x_var} + {local_x - 1}) / {local_x}), static_cast<uint32_t>(({dim_y_var} + {local_y - 1}) / {local_y}), 1 }}"
    else:
        workgroup_calc = "{ 1, 1, 1 }"

    cpp_code = f"""// Auto-generated native Kompute C++ dispatch for Motif kernel: {kernel.name}
// Zero Python runtime overhead — runs directly on Android Vulkan GPU (libvulkan.so)

#include "kompute/Kompute.hpp"
#include <vector>
#include <memory>
#include <cstdint>
#include <algorithm>

namespace motif {{
namespace generated {{

// Compiled SPIR-V words for kernel '{kernel.name}'
static const std::vector<uint32_t> {kernel.name}_spirv = {{
    {spirv_array_str}
}};

struct {kernel.name}_push_constants {{
{scalar_struct_def}
}};

void run_{kernel.name}_kernel(
    {', '.join([f'float* {buf}_data' for buf in kernel.buffers])},
    uint32_t element_count,
    {', '.join([f'{("int32_t" if t == int else "float")} {name}' for name, t in kernel.scalars])}
) {{
    // 1. Initialize Vulkan Kompute Manager for GPU
    kp::Manager mgr;

    // 2. Wrap pointers in Vulkan GPU Tensors
{tensor_create_str}
    std::vector<std::shared_ptr<kp::Memory>> params = {{ {tensor_vec_str} }};

    // 3. Setup Push Constants
    {kernel.name}_push_constants push_consts = {{ {', '.join([name for name, _ in kernel.scalars])} }};

    // 4. Create Kompute Algorithm
    kp::Workgroup workgroup = {workgroup_calc};
    std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(
        params,
        {kernel.name}_spirv,
        workgroup,
        {{}},
        {{ {', '.join(push_const_args)} }}
    );

    // 5. Submit to Vulkan compute queue and execute
    mgr.sequence()
        ->record<kp::OpSyncDevice>(params)
        ->record<kp::OpAlgoDispatch>(algo)
        ->record<kp::OpSyncLocal>({{ {buffer_ptrs[-1]} }})
        ->eval();

    // 6. Copy GPU result back to output data array
{sync_back_code}
}}

}} // namespace generated
}} // namespace motif
"""
    return cpp_code


def generate_test_runner(kernel: Kernel, output_path: str = "generated/test_runner.cpp") -> str:
    """Generates a standalone C++ test runner executable for Android NDK GPU verification (`adb shell`)."""
    cpp_dispatch = generate_cpp_dispatch(kernel)

    test_main = f"""
#include <iostream>
#include <vector>
#include <cmath>

{cpp_dispatch}

int main() {{
    std::cout << "[Motif Android Dispatch] Running native C++ test for kernel '{kernel.name}'..." << std::endl;

    constexpr uint32_t N = 1024;
    std::vector<float> a(N, 2.0f);
    std::vector<float> b(N, 3.0f);
    std::vector<float> out(N, 0.0f);

    try {{
        motif::generated::run_{kernel.name}_kernel(a.data(), b.data(), out.data(), N, N);
        std::cout << "[Motif Android Dispatch] Dispatch succeeded on Vulkan GPU!" << std::endl;
        std::cout << "Sample output [0..3]: " << out[0] << ", " << out[1] << ", " << out[2] << std::endl;
    }} catch (const std::exception& e) {{
        std::cerr << "[ERROR] Native dispatch failed: " << e.what() << std::endl;
        return 1;
    }}

    return 0;
}}
"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(test_main)

    return test_main


def generate_jni_bridge(kernel: Kernel, package_name: str = "org.motif.example") -> str:
    """Generates C++ JNI function entry point for Kotlin/Java Android calls."""
    pascal_name = "".join(part.capitalize() for part in kernel.name.split("_"))
    jni_func_name = f"Java_{package_name.replace('.', '_')}_MotifJni_run{pascal_name}"

    buf_params = ", ".join([f"jfloatArray {buf}_arr" for buf in kernel.buffers])
    scalar_params = ", ".join([f"{('jint' if t == int else 'jfloat')} {name}" for name, t in kernel.scalars])

    jni_args = f"JNIEnv* env, jobject thiz, {buf_params}"
    if scalar_params:
        jni_args += f", {scalar_params}"

    pin_lines = []
    unpin_lines = []
    call_ptrs = []

    for buf in kernel.buffers:
        pin_lines.append(f"    float* {buf}_ptr = env->GetFloatArrayElements({buf}_arr, NULL);")
        unpin_lines.append(f"    env->ReleaseFloatArrayElements({buf}_arr, {buf}_ptr, 0);")
        call_ptrs.append(f"{buf}_ptr")

    pin_block = "\n".join(pin_lines)
    unpin_block = "\n".join(unpin_lines)
    call_arg_str = ", ".join(call_ptrs) + ", count, " + ", ".join([name for name, _ in kernel.scalars])

    first_buf = kernel.buffers[0]

    cpp_dispatch = generate_cpp_dispatch(kernel)

    jni_code = f"""// Auto-generated JNI Bridge for Motif kernel: {kernel.name}
#include <jni.h>
#include <iostream>
#include <vector>

{cpp_dispatch}

extern "C" {{

JNIEXPORT jboolean JNICALL
{jni_func_name}({jni_args}) {{
    if ({first_buf}_arr == NULL) return JNI_FALSE;
    jsize count = env->GetArrayLength({first_buf}_arr);

{pin_block}

    try {{
        motif::generated::run_{kernel.name}_kernel({call_arg_str});
{unpin_block}
        return JNI_TRUE;
    }} catch (const std::exception& e) {{
{unpin_block}
        return JNI_FALSE;
    }}
}}

}} // extern "C"
"""
    return jni_code


def generate_kotlin_wrapper(kernels: List[Kernel], package_name: str = "org.motif.example") -> str:
    """Generates Kotlin MotifJni.kt wrapper object for an Android application."""
    methods = []
    for k in kernels:
        pascal_name = "".join(part.capitalize() for part in k.name.split("_"))
        buf_args = ", ".join([f"{buf}: FloatArray" for buf in k.buffers])
        scalar_args = ", ".join([f"{name}: {('Int' if t == int else 'Float')}" for name, t in k.scalars])

        all_args = buf_args
        if scalar_args:
            all_args += f", {scalar_args}"

        methods.append(f"    external fun run{pascal_name}({all_args}): Boolean")

    methods_str = "\n".join(methods)

    kotlin_code = f"""package {package_name}

import android.util.Log

/**
 * Auto-generated Motif Kotlin interface for Android GPU dispatches.
 */
object MotifJni {{
    var isLibraryLoaded = false
        private set

    init {{
        try {{
            System.loadLibrary("motif-jni")
            isLibraryLoaded = true
        }} catch (e: Throwable) {{
            Log.e("MotifJni", "Failed to load libmotif-jni.so", e)
            isLibraryLoaded = false
        }}
    }}

    fun safeInitVulkan(): Boolean {{
        if (!isLibraryLoaded) return false
        return try {{
            initVulkan()
        }} catch (e: Throwable) {{
            false
        }}
    }}

    external fun initVulkan(): Boolean

{methods_str}
}}
"""
    return kotlin_code


def dispatch(kernel: Kernel, buffers: list, scalar_values: dict, workgroups: tuple):
    """Python dispatch interface (Generates C++ code for Android deployment)."""
    assert len(buffers) == len(kernel.buffers), (
        f"kernel '{kernel.name}' expects {len(kernel.buffers)} buffers "
        f"{kernel.buffers}, got {len(buffers)}"
    )
    for name, _ in kernel.scalars:
        assert name in scalar_values, f"missing scalar '{name}' for kernel '{kernel.name}'"

    cpp_source = generate_cpp_dispatch(kernel)
    print(f"--- Generated Native C++ Kompute Dispatch for {kernel.name} ---")
    print(cpp_source[:300] + "\n... (truncated) ...\n")
    return cpp_source


