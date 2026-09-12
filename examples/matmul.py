import motif
from motif import Buffer


@motif.kernel
def matmul(a: Buffer, b: Buffer, c: Buffer, M: int, N: int, K: int):
    row = motif.global_id(0)
    col = motif.global_id(1)
    if row < M and col < N:
        acc = 0.0
        for k in range(K):
            acc += a[row * K + k] * b[k * N + col]
        c[row * N + col] = acc


if __name__ == "__main__":
    print(matmul.glsl)
    print("dispatch_dims:", matmul.dispatch_dims, "local_size:", matmul.local_size)
    print("buffers:", matmul.buffers, "scalars:", matmul.scalars)
