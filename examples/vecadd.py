import motif
from motif import Buffer


@motif.kernel
def vecadd(a: Buffer, b: Buffer, out: Buffer, n: int):
    i = motif.global_id(0)
    if i < n:
        out[i] = a[i] + b[i]


if __name__ == "__main__":
    print(vecadd.glsl)
    print("dispatch_dims:", vecadd.dispatch_dims, "local_size:", vecadd.local_size)
    print("buffers:", vecadd.buffers, "scalars:", vecadd.scalars)
