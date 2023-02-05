import tvm
import tvm.testing
from tvm import te
import numpy as np
print('here')

tgt_gpu = tvm.target.Target(target='cuda', host='llvm')

n = te.var("n")
A = te.placeholder((n, ), name='A')
B = te.placeholder((n, ), name="B")
C = te.compute(A.shape, lambda i : A[i] + B[i], name="C")
print(type(C))

s = te.create_schedule(C.op)

bx, tx = s[C].split(C.op.axis[0], factor=64)

s[C].bind(bx, te.thread_axis("blockIdx.x"))
s[C].bind(tx, te.thread_axis("threadIdx.x"))

fadd = tvm.build(s, [A, B, C], target=tgt_gpu, name='myadd')

dev = tvm.device(tgt_gpu.kind.name, 0)

n=1024
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
fadd(a, b, c)
tvm.testing.assert_allclose(c.numpy(), a.numpy()+b.numpy())

if(tgt_gpu.kind.name=='cuda'
    or tgt_gpu.kind.name=='rocm'
    or tgt_gpu.kind.name.startwith('opencl')):
    dev_module = fadd.imported_modules[0]
    print("-----GPU code-----")
    print(dev_module.get_source())
