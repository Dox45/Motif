package org.motif.example

/**
 * High-level execution engine exposing GPU-accelerated operations on MotifTensors.
 */
object MotifEngine {

    /**
     * GPU Elementwise Vector Addition: out = a + b
     */
    fun vecadd(a: MotifTensor, b: MotifTensor, out: MotifTensor): Boolean {
        require(a.size == b.size && a.size == out.size) { "Tensor dimensions must match for vecadd" }
        return MotifJni.runVecadd(a.data, b.data, out.data, a.size)
    }

    /**
     * GPU Matrix Multiplication: C = A x B
     * Tensor A shape: [M, K]
     * Tensor B shape: [K, N]
     * Tensor C shape: [M, N]
     */
    fun matmul(a: MotifTensor, b: MotifTensor, c: MotifTensor): Boolean {
        require(a.rank == 2 && b.rank == 2 && c.rank == 2) { "Tensors must be 2D matrices for matmul" }
        val M = a.shape[0]
        val K = a.shape[1]
        val N = b.shape[1]

        require(b.shape[0] == K) { "Matrix A cols ($K) must match Matrix B rows (${b.shape[0]})" }
        require(c.shape[0] == M && c.shape[1] == N) { "Output Matrix C shape must be [$M, $N]" }

        return MotifJni.runMatmul(a.data, b.data, c.data, M, N, K)
    }
}
