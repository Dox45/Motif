#!/usr/bin/env python3
"""
Motif Android Kernel Compiler Script

Compiles Python & GLSL kernels (`vecadd`, `matmul`, `logistic`) into C++ JNI bridge files
and Kotlin wrappers for the Motif Android Example App.
"""

import sys
import os
import subprocess
import struct

# Include parent directory in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from examples.vecadd import vecadd
from examples.matmul import matmul
from motif.dispatch import generate_jni_bridge, generate_kotlin_wrapper

def compile_glsl_to_spirv_words(glsl_path: str) -> list:
    """Compiles GLSL shader to SPIR-V words using glslc."""
    spv_temp = "/tmp/motif_temp.spv"
    try:
        subprocess.run(["glslc", glsl_path, "-o", spv_temp], check=True)
        data = open(spv_temp, "rb").read()
        words = [f"0x{w:08x}" for w, in struct.iter_unpack("<I", data)]
        return words
    finally:
        if os.path.exists(spv_temp):
            os.unlink(spv_temp)

def generate_logistic_cpp(glsl_path: str, cpp_path: str):
    words = compile_glsl_to_spirv_words(glsl_path)
    spirv_str = ",\n    ".join([", ".join(words[i:i+8]) for i in range(0, len(words), 8)])
    
    cpp_code = f"""// Auto-generated native Kompute C++ dispatch for Motif Logistic Regression kernel
#include "kompute/Kompute.hpp"
#include <vector>
#include <memory>
#include <cstdint>
#include <algorithm>
#include <jni.h>
#include <cmath>

namespace motif {{
namespace generated {{

static const std::vector<uint32_t> logistic_spirv = {{
    {spirv_str}
}};

bool run_logistic_step_gpu(
    float* xi, float* xj, float* y,
    float* win, float* wouti, float* woutj,
    float* bin, float* bout, float* lout,
    uint32_t element_count
) {{
    try {{
        kp::Manager mgr;
        std::shared_ptr<kp::TensorT<float>> t_xi = mgr.tensor(std::vector<float>(xi, xi + element_count));
        std::shared_ptr<kp::TensorT<float>> t_xj = mgr.tensor(std::vector<float>(xj, xj + element_count));
        std::shared_ptr<kp::TensorT<float>> t_y = mgr.tensor(std::vector<float>(y, y + element_count));
        std::shared_ptr<kp::TensorT<float>> t_win = mgr.tensor(std::vector<float>(win, win + 2));
        std::shared_ptr<kp::TensorT<float>> t_wouti = mgr.tensor(std::vector<float>(wouti, wouti + element_count));
        std::shared_ptr<kp::TensorT<float>> t_woutj = mgr.tensor(std::vector<float>(woutj, woutj + element_count));
        std::shared_ptr<kp::TensorT<float>> t_bin = mgr.tensor(std::vector<float>(bin, bin + 1));
        std::shared_ptr<kp::TensorT<float>> t_bout = mgr.tensor(std::vector<float>(bout, bout + element_count));
        std::shared_ptr<kp::TensorT<float>> t_lout = mgr.tensor(std::vector<float>(lout, lout + element_count));

        std::vector<std::shared_ptr<kp::Memory>> params = {{
            t_xi, t_xj, t_y, t_win, t_wouti, t_woutj, t_bin, t_bout, t_lout
        }};

        kp::Workgroup workgroup = {{ element_count, 1, 1 }};
        std::vector<float> specConsts = {{ static_cast<float>(element_count) }};
        std::shared_ptr<kp::Algorithm> algo = mgr.algorithm(
            params,
            logistic_spirv,
            workgroup,
            specConsts,
            {{}}
        );

        mgr.sequence()
            ->record<kp::OpSyncDevice>(params)
            ->record<kp::OpAlgoDispatch>(algo)
            ->record<kp::OpSyncLocal>({{ t_wouti, t_woutj, t_bout, t_lout }})
            ->eval();

        std::vector<float> r_wouti = t_wouti->vector();
        std::vector<float> r_woutj = t_woutj->vector();
        std::vector<float> r_bout = t_bout->vector();
        std::vector<float> r_lout = t_lout->vector();

        std::copy(r_wouti.begin(), r_wouti.end(), wouti);
        std::copy(r_woutj.begin(), r_woutj.end(), woutj);
        std::copy(r_bout.begin(), r_bout.end(), bout);
        std::copy(r_lout.begin(), r_lout.end(), lout);

        return true;
    }} catch (...) {{
        // CPU fallback implementation for devices/emulators without Vulkan compute support
        float w1 = win[0];
        float w2 = win[1];
        float b = bin[0];
        float m = static_cast<float>(element_count);
        for (size_t i = 0; i < element_count; i++) {{
            float z = w1 * xi[i] + w2 * xj[i] + b;
            float yHat = 1.0f / (1.0f + std::exp(-z));
            float dZ = yHat - y[i];
            wouti[i] = (1.0f / m) * xi[i] * dZ;
            woutj[i] = (1.0f / m) * xj[i] * dZ;
            bout[i] = (1.0f / m) * dZ;
            float safeYhat = std::max(1e-7f, std::min(1.0f - 1e-7f, yHat));
            lout[i] = -(y[i] * std::log(safeYhat) + (1.0f - y[i]) * std::log(1.0f - safeYhat));
        }}
        return true;
    }}
}}

}} // namespace generated
}} // namespace motif

extern "C" {{
JNIEXPORT jboolean JNICALL
Java_org_motif_example_MotifJni_runLogisticStep(
    JNIEnv* env, jobject thiz,
    jfloatArray xi, jfloatArray xj, jfloatArray y,
    jfloatArray win, jfloatArray wouti, jfloatArray woutj,
    jfloatArray bin, jfloatArray bout, jfloatArray lout
) {{
    if (!xi || !xj || !y || !win || !wouti || !woutj || !bin || !bout || !lout) return JNI_FALSE;
    jsize n = env->GetArrayLength(xi);

    float* p_xi = env->GetFloatArrayElements(xi, NULL);
    float* p_xj = env->GetFloatArrayElements(xj, NULL);
    float* p_y = env->GetFloatArrayElements(y, NULL);
    float* p_win = env->GetFloatArrayElements(win, NULL);
    float* p_wouti = env->GetFloatArrayElements(wouti, NULL);
    float* p_woutj = env->GetFloatArrayElements(woutj, NULL);
    float* p_bin = env->GetFloatArrayElements(bin, NULL);
    float* p_bout = env->GetFloatArrayElements(bout, NULL);
    float* p_lout = env->GetFloatArrayElements(lout, NULL);

    bool res = motif::generated::run_logistic_step_gpu(
        p_xi, p_xj, p_y, p_win, p_wouti, p_woutj, p_bin, p_bout, p_lout, n
    );

    env->ReleaseFloatArrayElements(xi, p_xi, 0);
    env->ReleaseFloatArrayElements(xj, p_xj, 0);
    env->ReleaseFloatArrayElements(y, p_y, 0);
    env->ReleaseFloatArrayElements(win, p_win, 0);
    env->ReleaseFloatArrayElements(wouti, p_wouti, 0);
    env->ReleaseFloatArrayElements(woutj, p_woutj, 0);
    env->ReleaseFloatArrayElements(bin, p_bin, 0);
    env->ReleaseFloatArrayElements(bout, p_bout, 0);
    env->ReleaseFloatArrayElements(lout, p_lout, 0);

    return res ? JNI_TRUE : JNI_FALSE;
}}
}}
"""
    with open(cpp_path, "w") as f:
        f.write(cpp_code)
    print(f"[Motif Android Compiler] Generated C++ Logistic kernel: {cpp_path}")

def main():
    base_dir = os.path.dirname(__file__)
    cpp_out_dir = os.path.join(base_dir, "app/src/main/cpp/generated")
    kotlin_out_dir = os.path.join(base_dir, "app/src/main/java/org/motif/example")

    os.makedirs(cpp_out_dir, exist_ok=True)
    os.makedirs(kotlin_out_dir, exist_ok=True)

    kernels = [vecadd, matmul]

    for k in kernels:
        cpp_code = generate_jni_bridge(k, package_name="org.motif.example")
        cpp_path = os.path.join(cpp_out_dir, f"motif_{k.name}.cpp")
        with open(cpp_path, "w") as f:
            f.write(cpp_code)
        print(f"[Motif Android Compiler] Generated JNI C++ bridge: {cpp_path}")

    glsl_logistic = os.path.abspath(os.path.join(base_dir, "../../third_party/kompute/src/shaders/glsl/ShaderLogisticRegression.comp"))
    cpp_logistic = os.path.join(cpp_out_dir, "motif_logistic.cpp")
    generate_logistic_cpp(glsl_logistic, cpp_logistic)

    kotlin_code = generate_kotlin_wrapper(kernels, package_name="org.motif.example")
    kotlin_path = os.path.join(kotlin_out_dir, "MotifJni.kt")
    with open(kotlin_path, "w") as f:
        f.write(kotlin_code)
    print(f"[Motif Android Compiler] Generated Kotlin Interface: {kotlin_path}")

if __name__ == "__main__":
    main()

