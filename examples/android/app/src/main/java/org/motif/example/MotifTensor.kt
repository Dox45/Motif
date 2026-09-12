package org.motif.example

/**
 * High-level Android Kotlin Tensor abstraction for Motif GPU dispatches.
 */
class MotifTensor(
    val data: FloatArray,
    val shape: IntArray
) {
    val size: Int
        get() = data.size

    val rank: Int
        get() = shape.size

    companion object {
        fun fromArray(array: FloatArray, shape: IntArray): MotifTensor {
            var expectedSize = 1
            for (dim in shape) {
                expectedSize *= dim
            }
            require(array.size == expectedSize) {
                "Array size (${array.size}) does not match shape ${shape.contentToString()} (expected $expectedSize)"
            }
            return MotifTensor(array, shape)
        }

        fun allocate(shape: IntArray): MotifTensor {
            var size = 1
            for (dim in shape) {
                size *= dim
            }
            return MotifTensor(FloatArray(size), shape)
        }
    }

    fun toArray(): FloatArray {
        return data.clone()
    }
}
