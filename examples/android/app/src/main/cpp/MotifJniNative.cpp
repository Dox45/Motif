#include <jni.h>
#include <kompute/Kompute.hpp>
#include <exception>

extern "C" {

JNIEXPORT jboolean JNICALL
Java_org_motif_example_MotifJni_initVulkan(JNIEnv* env, jobject thiz) {
    try {
        kp::Manager mgr;
        return JNI_TRUE;
    } catch (const std::exception& e) {
        return JNI_FALSE;
    } catch (...) {
        return JNI_FALSE;
    }
}

}
