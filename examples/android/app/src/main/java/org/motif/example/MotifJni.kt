package org.motif.example

import android.util.Log

/**
 * Auto-generated Motif Kotlin interface for Android GPU dispatches.
 */
object MotifJni {
    var isLibraryLoaded = false
        private set

    init {
        try {
            System.loadLibrary("motif-jni")
            isLibraryLoaded = true
        } catch (e: Throwable) {
            Log.e("MotifJni", "Failed to load libmotif-jni.so", e)
            isLibraryLoaded = false
        }
    }

    fun safeInitVulkan(): Boolean {
        if (!isLibraryLoaded) return false
        return try {
            initVulkan()
        } catch (e: Throwable) {
            false
        }
    }

    external fun initVulkan(): Boolean

    external fun runVecadd(a: FloatArray, b: FloatArray, out: FloatArray, n: Int): Boolean
    external fun runMatmul(a: FloatArray, b: FloatArray, c: FloatArray, M: Int, N: Int, K: Int): Boolean

    external fun runLogisticStep(
        xi: FloatArray, xj: FloatArray, y: FloatArray,
        win: FloatArray, wouti: FloatArray, woutj: FloatArray,
        bin: FloatArray, bout: FloatArray, lout: FloatArray
    ): Boolean

    fun trainLogisticModel(
        xi: FloatArray, xj: FloatArray, y: FloatArray,
        iterations: Int = 100, learningRate: Float = 0.1f
    ): Triple<FloatArray, Float, Float> {
        val n = xi.size
        val win = floatArrayOf(0.0f, 0.0f)
        val bin = floatArrayOf(0.0f)

        val wouti = FloatArray(n)
        val woutj = FloatArray(n)
        val bout = FloatArray(n)
        val lout = FloatArray(n)

        var lastLoss = 0.0f

        for (iter in 0 until iterations) {
            val stepOk = try {
                runLogisticStep(xi, xj, y, win, wouti, woutj, bin, bout, lout)
            } catch (e: Throwable) {
                false
            }

            if (!stepOk) {
                // Kotlin CPU Fallback
                for (i in 0 until n) {
                    val z = win[0] * xi[i] + win[1] * xj[i] + bin[0]
                    val yHat = 1.0f / (1.0f + kotlin.math.exp(-z))
                    val dZ = yHat - y[i]
                    wouti[i] = (1.0f / n) * xi[i] * dZ
                    woutj[i] = (1.0f / n) * xj[i] * dZ
                    bout[i] = (1.0f / n) * dZ
                    val safeYhat = kotlin.math.max(1e-7f, kotlin.math.min(1.0f - 1e-7f, yHat))
                    lout[i] = -(y[i] * kotlin.math.ln(safeYhat) + (1.0f - y[i]) * kotlin.math.ln(1.0f - safeYhat))
                }
            }

            var sumDw1 = 0.0f
            var sumDw2 = 0.0f
            var sumDb = 0.0f
            var sumLoss = 0.0f
            for (i in 0 until n) {
                sumDw1 += wouti[i]
                sumDw2 += woutj[i]
                sumDb += bout[i]
                sumLoss += lout[i]
            }

            win[0] -= learningRate * sumDw1
            win[1] -= learningRate * sumDw2
            bin[0] -= learningRate * sumDb
            lastLoss = sumLoss / n
        }

        return Triple(win, bin[0], lastLoss)
    }
}

