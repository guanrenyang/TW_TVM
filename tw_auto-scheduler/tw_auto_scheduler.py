import tvm
import tvm.testing
from tvm import te, auto_scheduler
import numpy as np
import random

from extern_func import insert_row_to_C_kernel

# N_pruned is the number of remaining entries in N dimension
# TODO: change K_pruned to a layer-wise configuration
big_test = True
dtype = 'float32'
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
block_num = (N_pruned_global + tile_size - 1)//tile_size
N_ori_per_block = N // block_num

@auto_scheduler.register_workload
def tw_kernel(M, K, K_pruned_max, tile_size, block_num, dtype='float32'):

    A_transposed = te.placeholder((K, M), name='A_transposed', dtype=dtype)
    B_transposed_packed = te.placeholder((block_num, tile_size, K_pruned_max), name='B_transposed_packed', dtype=dtype)
    
    mask_k = te.placeholder((block_num, K_pruned_max), name='mask_k', dtype='int32')

    A_transposed_skipped = te.compute((block_num, K_pruned_max, M), lambda bn, i, j: A_transposed[mask_k[bn, i], j], name='A_transposed_skipped', tag='sparse_read_from_A')
    
    k = te.reduce_axis((0, K_pruned_max), name='k')
    C_transposed_skipped = te.compute((block_num, tile_size, M), lambda bn, j, i: te.sum(A_transposed_skipped[bn, k, i] * B_transposed_packed[bn, j, k].astype(dtype), axis=k) , name='C_transposed_skipped')

    return [C_transposed_skipped, A_transposed, B_transposed_packed, mask_k]
    
# generate random mask_k for Ansor
mask_k_test = np.zeros((block_num, K_pruned_max)).astype('int32')
for row in range(block_num):
    mask_k_test[row, :] = np.random.choice(K, K_pruned_max, replace=False)
mask_k_test.sort(axis=1)

# create search task
target = tvm.target.cuda(arch='sm_70')
device = tvm.device(target.get_target_device_type(), 0)

task = tvm.auto_scheduler.SearchTask(func=tw_kernel, args=(M, K, K_pruned_max, tile_size, block_num, dtype), target=target, 
                                     task_inputs={'mask_k': tvm.runtime.ndarray.array(mask_k_test)})

# Inspect the computational graph
print("Computational DAG:")
print(task.compute_dag)

# Custom Sketch
def meet_condition_func(search_policy, state, stage_id):
    state = auto_scheduler.loop_state.State(state, search_policy.search_task.compute_dag)
    print(stage_id, state.stages[stage_id].op.tag)
    if state.stages[stage_id].op.tag in [
        "A_transposed_skipped",
    ]:
        return auto_scheduler.PreloadCustomSketchRule.APPLY_AND_SKIP_REST
    else:
        return auto_scheduler.PreloadCustomSketchRule.PASS


def apply_func(search_policy, state, stage_id):
    ret = []

    return ret

# Set parameters for auto_scheduler
log_file = "tw.json"
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=10,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)

search_policy = auto_scheduler.SketchPolicy(
    task,
    program_cost_model=auto_scheduler.XGBModel(),
    init_search_callbacks=[
        auto_scheduler.PreloadCustomSketchRule(meet_condition_func, apply_func)
    ],
)

# Run auto-tuning (search)
task.tune(tune_option)

# Apply the best schedule
sch_tw_mainbody, args_tw_mainbody = task.apply_best(log_file)
# Get the extern function `coowrite`
sch_tw_coowrite_C, args_tw_coowrite_C = insert_row_to_C_kernel(M, N, tile_size, block_num, dtype)

# # Inspecting the Optimized Schedule
# print("Lowered TIR of tw mainbody:")
# print(tvm.lower(sch_tw_mainbody, args_tw_mainbody, simple_mode=True))
# print("Lowered TIR of coowrite")
# print(tvm.lower(sch_tw_coowrite_C, args_tw_coowrite_C, simple_mode=True))

# Check correctness and evaluate performance
tw_mainbody = tvm.build(sch_tw_mainbody, args_tw_mainbody, target)
tw_coowrite_C = tvm.build(sch_tw_coowrite_C, args_tw_coowrite_C, target)

'''generate data for testing'''
# generate mask_n
mask_n_test = np.zeros((block_num, tile_size)).astype('int32')
for row in range(block_num):
    mask_n_test[row, :] = np.random.choice(N_ori_per_block, tile_size, replace=False)
mask_n_test.sort(axis=1)

A_transposed_test = np.random.random((K, M)).astype(dtype)
B_transposed_test = np.random.random((N, K)).astype(dtype)
C_transposed_skipped_test = np.zeros((block_num, tile_size, M)).astype(dtype)
C_transposed_test = np.zeros((N, M)).astype(dtype)


# apply mask to B_transposed
B_transposed_pruned = np.zeros((N, K)).astype(dtype)
for bn in range(block_num):
    # print(bn)
    for ts in range(tile_size):
        for k in range(K_pruned_max):
            B_transposed_pruned[(bn*N_ori_per_block+mask_n_test[bn, ts]), mask_k_test[bn, k]] = B_transposed_test[(bn*N_ori_per_block+mask_n_test[bn, ts]), mask_k_test[bn, k]]

# transpose B to B_packed
B_transposed_packed_test = np.zeros((block_num, tile_size, K_pruned_max)).astype(dtype)
for bn in range(block_num):
    # print(bn)
    for ts in range(tile_size):
        for k in range(K_pruned_max):
            B_transposed_packed_test[bn, ts, k] = B_transposed_pruned[(bn*N_ori_per_block+mask_n_test[bn, ts]), mask_k_test[bn, k]]


A_transposed_test = tvm.nd.array(A_transposed_test, device)
B_transposed_packed_test = tvm.nd.array(B_transposed_packed_test, device)
C_transposed_skipped_test = tvm.nd.array(C_transposed_skipped_test, device) # a temporary buffer
C_transposed_test = tvm.nd.array(C_transposed_test, device)
mask_k_test = tvm.nd.array(mask_k_test, device)
mask_n_test = tvm.nd.array(mask_n_test, device)

# run the kernel
tw_mainbody(C_transposed_skipped_test, A_transposed_test, B_transposed_packed_test, mask_k_test)
tw_coowrite_C(C_transposed_test, C_transposed_skipped_test, mask_n_test)

tvm.testing.assert_allclose(C_transposed_test.numpy().T, A_transposed_test.numpy().T @ B_transposed_pruned.T, 1e-1)



