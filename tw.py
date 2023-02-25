import tvm
import tvm.testing
from tvm import te, auto_scheduler
import numpy as np
import random

# tgt_gpu = tvm.target.cuda()
# gpu_0 = tvm.device(tgt_gpu.kind.name, 0)

# tgt_cpu = tvm.target.Target(target="llvm", host="llvm")
# cpu = tvm.device(tgt_cpu.kind.name, 0)

# N_pruned is the number of remaining entries in N dimension
# TODO: change K_pruned to a layer-wise configuration
big_test = True
if big_test:
    '''test for accuracy'''
    M = 1024
    N = 1024
    K = 1024
    tile_size = 32
    K_pruned_max = 512
    N_pruned_global = 512
else:
    '''test for visualization'''
    M = 16
    N = 16
    K = 16
    tile_size =  2
    K_pruned_max = 8
    N_pruned_global = 8



#int, N: int, K:int, K_pruned_max: int, N_pruned:int, tile_size:int,
def get_tw_kernel( M: int, N: int, K:int, K_pruned_max: int, N_pruned_global:int, tile_size:int, cuda:bool=False, enable_test:str='cpu'):
    '''TW Tiled-Gemm kernel
    Input of the kernel: 
    * A: K*M
    * B: (block_num, tile_size, K_pruned_max)
    * C: N*M
    * mask_k: (block_num, K_pruned_max)
    * mask_n: (block_num, tile_size)
    * block_num
    Output of the kernel:
    * '''
    dtype = 'float16'
    block_num = (N_pruned_global + tile_size - 1)//tile_size
    print('block_num', block_num)
    N_ori_per_block = N // block_num

    A_transposed = te.placeholder((K, M), name='A_transposed', dtype=dtype)
    B_transposed_packed = te.placeholder((block_num, tile_size, K_pruned_max), name='B_transposed_packed', dtype=dtype)
    

    mask_k = te.placeholder((block_num, K_pruned_max), name='mask_k', dtype='int')
    mask_n = te.placeholder((block_num, tile_size), name='mask_n', dtype='int') 

    A_transposed_skipped = te.compute((block_num, K_pruned_max, M), lambda bn, i, j: A_transposed[mask_k[bn, i], j], name='A_transposed_skipped')
    
    k = te.reduce_axis((0, K_pruned_max), name='k')
    C_transposed_skipped = te.compute((block_num, tile_size, M), lambda bn, j, i: te.sum(A_transposed_skipped[bn, k, i] * B_transposed_packed[bn, j, k].astype(dtype), axis=k) , name='C_transposed_skipped')

    def write_C_to_sparse(data, mask_n, out):
        '''
        data: shape of (block_num, tile_size, M)
        mask_n: shape of (block_num, tile_size)
        '''
        irb = tvm.tir.ir_builder.create()
        data_ptr = irb.buffer_ptr(data)
        mask_n_ptr = irb.buffer_ptr(mask_n)
        out_ptr = irb.buffer_ptr(out)

        assert data.shape[0]==mask_n.shape[0], 'block_num mismatches'
        block_num = data.shape[0]
        assert data.shape[1]==mask_n.shape[1], 'tile_size mismatches'
        tile_size = data.shape[1]
        
        N = out.shape[0]
        M = out.shape[1]

        N_ori_per_block = N // block_num

        with irb.for_range(0, N, kind='serial', name='n') as n:
            with irb.for_range(0, M, kind='serial', name='m') as m:
                out_ptr[n * M + m] = tvm.tir.generic.cast(0, data.dtype)

        with irb.for_range(0, block_num, kind='serial', name='bn') as bn:
            with irb.for_range(0, tile_size, kind='serial', name='ts') as ts:
                with irb.for_range(0, M, kind='serial', name='col') as col:
                    out_ptr[(N_ori_per_block * bn + mask_n_ptr[bn * tile_size + ts]) * M + col] += data_ptr[bn * tile_size * M + ts * M + col]
        return irb.get()
        
    C_transposed = te.extern((N, M),
                             [C_transposed_skipped, mask_n],
                             lambda ins, outs: write_C_to_sparse(ins[0], ins[1], outs[0]),
                             tag='write_C_to_sparse',
                             dtype=C_transposed_skipped.dtype,
                             name='C_transposed',
                             )
    
    s = te.create_schedule(C_transposed.op)

    if enable_test == 'cpu':
        '''testing cpu'''
        tw_kernel = tvm.build(s, [C_transposed, A_transposed, B_transposed_packed, mask_k, mask_n], tgt_cpu, name='tiled_matmul')

        # generate mask_n and mask_k
        mask_n_test = np.zeros((block_num, tile_size)).astype(mask_n.dtype)
        for row in range(block_num):
            mask_n_test[row, :] = np.random.choice(N_ori_per_block, tile_size, replace=False)
        mask_n_test.sort(axis=1)

        mask_k_test = np.zeros((block_num, K_pruned_max)).astype(mask_k.dtype)
        for row in range(block_num):
            mask_k_test[row, :] = np.random.choice(K, K_pruned_max, replace=False)
        mask_k_test.sort(axis=1)

        A_transposed_test = np.random.random((K, M)).astype(A_transposed.dtype)
        B_transposed_test = np.random.random((N, K)).astype(B_transposed_packed.dtype)
        C_transposed_test = np.zeros((N, M)).astype(C_transposed.dtype)

        
        # apply mask to B_transposed
        B_transposed_pruned = np.zeros((N, K)).astype(B_transposed_packed.dtype)
        for bn in range(block_num):
            # print(bn)
            for ts in range(tile_size):
                for k in range(K_pruned_max):
                    B_transposed_pruned[(bn*N_ori_per_block+mask_n_test[bn, ts]), mask_k_test[bn, k]] = B_transposed_test[(bn*N_ori_per_block+mask_n_test[bn, ts]), mask_k_test[bn, k]]
    
        # transpose B to B_packed
        B_transposed_packed_test = np.zeros((block_num, tile_size, K_pruned_max)).astype(B_transposed_packed.dtype)
        for bn in range(block_num):
            # print(bn)
            for ts in range(tile_size):
                for k in range(K_pruned_max):
                    B_transposed_packed_test[bn, ts, k] = B_transposed_pruned[(bn*N_ori_per_block+mask_n_test[bn, ts]), mask_k_test[bn, k]]

        A_transposed_test = tvm.nd.array(A_transposed_test)
        B_transposed_packed_test = tvm.nd.array(B_transposed_packed_test)
        C_transposed_test = tvm.nd.array(C_transposed_test)
        mask_k_test = tvm.nd.array(mask_k_test)
        mask_n_test = tvm.nd.array(mask_n_test)

        tw_kernel(C_transposed_test, A_transposed_test, B_transposed_packed_test, mask_k_test, mask_n_test)
        # print(C_transposed_test.numpy().T)
        # print("\n")
        # print( A_transposed_test.numpy().T @ B_transposed_pruned.T)
        tvm.testing.assert_allclose(C_transposed_test.numpy().T, A_transposed_test.numpy().T @ B_transposed_pruned.T, 1e-1)
    
    
    return s, [C_transposed, A_transposed, B_transposed_packed, mask_k, mask_n]
    # print(tvm.lower(s, [C_transposed, A_transposed, B_transposed_packed, mask_k, mask_n], simple_mode=True))
    
@auto_scheduler.register_workload
def tw_without_coowrite_kernel(M, K, K_pruned_max, N_pruned_global, tile_size, dtype='float16'):
    block_num = (N_pruned_global + tile_size - 1)//tile_size
    print('block_num', block_num)
    # N_ori_per_block = N // block_num

    A_transposed = te.placeholder((K, M), name='A_transposed', dtype=dtype)
    B_transposed_packed = te.placeholder((block_num, tile_size, K_pruned_max), name='B_transposed_packed', dtype=dtype)
    

    mask_k = te.placeholder((block_num, K_pruned_max), name='mask_k', dtype='int')
    mask_n = te.placeholder((block_num, tile_size), name='mask_n', dtype='int') 

    A_transposed_skipped = te.compute((block_num, K_pruned_max, M), lambda bn, i, j: A_transposed[mask_k[bn, i], j], name='A_transposed_skipped')
    
    k = te.reduce_axis((0, K_pruned_max), name='k')
    C_transposed_skipped = te.compute((block_num, tile_size, M), lambda bn, j, i: te.sum(A_transposed_skipped[bn, k, i] * B_transposed_packed[bn, j, k].astype(dtype), axis=k) , name='C_transposed_skipped')

    return [A_transposed, B_transposed_packed, mask_k, mask_n, C_transposed_skipped]
    
target = tvm.target.cuda(arch='sm_70')
task = tvm.auto_scheduler.SearchTask(func=tw_without_coowrite_kernel, args=(M, N, K, K_pruned_max, N_pruned_global, tile_size, 'float16'), target=target)

# Inspect the computational graph
print("Computational DAG:")
print(task.compute_dag)

# Set parameters for auto_scheduler
log_file = "tw_without_coowrite_kernel.json"
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=10,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)

# Run auto-tuning (search)
task.tune(tune_option)
# Apply the best schedule
sch, args = task.apply_best(log_file)

# Inspecting the Optimized Schedule
print("Lowered TIR:")
print(tvm.lower(sch, args, simple_mode=True))

# Check correctness and evaluate performance
func = tvm.build(sch, args, target)