import tvm
import tvm.testing
from tvm import te
import numpy as np
import random

tgt_gpu = tvm.target.Target(target="cuda", host="llvm")
gpu_0 = tvm.device(tgt_gpu.kind.name, 0)

tgt_cpu = tvm.target.Target(target="llvm", host="llvm")
cpu = tvm.device(tgt_cpu.kind.name, 0)

M = 1024
K = 2048
N = 1024
N_dim = 32
K_pruned = 128
N_pruned = 128

block_num = (N_pruned + N_dim - 1) // N_dim
N_ori = N//block_num

def mask_gen(N: int, N_pruned: int, base = 0):
    mask_keep = list(range(N))
    random.shuffle(mask_keep) # shuffle is an in-place operation
    mask_keep = mask_keep[ : N_pruned]
    mask_keep.sort()
    for i, _ in enumerate(mask_keep):
        mask_keep[i] = mask_keep[i] - i + base
    return mask_keep

def get_B_stream_and_masks(B, mask_k_list, mask_n_list):

    B_transposed_tiled_list = []
    for bn in range(block_num):
        mask_k = mask_k_list[bn]
        mask_n = mask_n_list[bn]
        
        dst = tvm.nd.array(np.zeros(N_dim, K_pruned))
        for i in range(K_pruned):
            for j in range(N_dim):
                idx_col = mask_k[i] + i
                idx_row = mask_n[j] + j + N_ori * bn
                dst[j, i] = B[idx_col, idx_row]
        B_transposed_tiled_list.append(dst)
    return B_transposed_tiled_list

def get_tiled_matmul_kernel():
    '''TW Tiled-Gemm
    Input: 
    * A_transposed
    * B_transposed_tiled
    * mask_k
    Output:
    * C_transposed_skipped'''
    A_transposed = te.placeholder((K, M), name='A_transposed')
    mask_k = te.placeholder((K_pruned,), name='mask_k', dtype='int')
    # mask_k = [0 for i in range(K_pruned)]
    # mask_n = te.placeholder((N_dim,), name="mask_n")

    A_transposed_skipped = te.compute((K_pruned, M), lambda i,j: A_transposed[i+mask_k[i], j], name='A_skipped')

    B_transposed_tiled = te.placeholder((N_dim, K_pruned), name='B_tiled')
    k = te.reduce_axis((0, K_pruned), name='k')
    C_transposed_skipped = te.compute((N_dim, M),lambda j,i: te.sum(A_transposed_skipped[k, i]*B_transposed_tiled[j, k], axis=k),name='C_transposed_skipped')

    s = te.create_schedule(C_transposed_skipped.op)

    tiled_matmul_kernel = tvm.build(s, [C_transposed_skipped, A_transposed, B_transposed_tiled, mask_k], target=tgt_cpu, name="tiled_matmul")

    print(tvm.lower(s, [C_transposed_skipped, A_transposed, B_transposed_tiled, mask_k], simple_mode=True))
    

    '''Testing'''
    A_transposed_data = tvm.nd.array(np.random.uniform(size=(K, M)).astype(A_transposed.dtype), cpu)
    B_transposed_tiled_data = tvm.nd.array(np.random.uniform(size=(N_dim, K_pruned)).astype(B_transposed_tiled.dtype), cpu)
    mask_k_data = tvm.nd.array(np.array(mask_gen(K, K_pruned)).astype(mask_k.dtype), cpu)
    # print(mask_k_data.dtype)
    C_transposed_skipped_data = tvm.nd.array(np.random.uniform(size=(N_dim, M)).astype(C_transposed_skipped.dtype), cpu)

    tiled_matmul_kernel(C_transposed_skipped_data, A_transposed_data, B_transposed_tiled_data, mask_k_data)

    def tiled_matmul_test(A_transposed, B_transposed_tiled, mask_k):
        A_transposed_skipped = np.zeros((K_pruned, M))
        for i in range(K_pruned):
            for j in range(M):
                A_transposed_skipped[i, j] = A_transposed[i+mask_k[i], j]
        C_transposed_skipped = (A_transposed_skipped.T @ B_transposed_tiled.T).T
        return C_transposed_skipped
    
    tvm.testing.assert_allclose(C_transposed_skipped_data.numpy(), tiled_matmul_test(A_transposed_data.numpy(), B_transposed_tiled_data.numpy(), mask_k_data.numpy()), 1e-6)
    '''End Testing'''
    return tiled_matmul_kernel

tiled_matmul_kernel = get_tiled_matmul_kernel()


