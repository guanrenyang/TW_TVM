import tvm
import tvm.testing
from tvm import te, auto_scheduler
import numpy as np
import random, os, json, math
# Some error occur when using relative import
from .extern_func import insert_row_to_C_kernel_tir  

is_resume = False
is_tune = True
ansor_num_trials = 2000
ansor_verbose = 0
root_dir = "/home/ryguan/TileSparsity_TVM/AutoSchedule/Outputs/"
latency_lookUpTable_dir =  os.path.join(root_dir, 'latency_lookup_table.json')

@auto_scheduler.register_workload
def tw_kernel(M, K, K_pruned_max, tile_size, block_num, dtype):

    A_transposed = te.placeholder((K, M), name='A_transposed', dtype=dtype)
    B_transposed_packed = te.placeholder((block_num, tile_size, K_pruned_max), name='B_transposed_packed', dtype=dtype)
    
    mask_k = te.placeholder((block_num, K_pruned_max), name='mask_k', dtype='int32')

    A_transposed_skipped = te.compute((block_num, K_pruned_max, M), lambda bn, i, j: A_transposed[mask_k[bn, i], j], name='A_transposed_skipped', tag='sparse_read_from_A')
    
    k = te.reduce_axis((0, K_pruned_max), name='k')
    C_transposed_skipped = te.compute((block_num, tile_size, M), lambda bn, j, i: te.sum(A_transposed_skipped[bn, k, i] * B_transposed_packed[bn, j, k].astype(dtype), axis=k) , name='C_transposed_skipped')

    return [C_transposed_skipped, A_transposed, B_transposed_packed, mask_k]
    
def add_to_lookUpTable(tile_size:int, sparsity:int, execution_time_list: list):
    tile_size = str(tile_size)
    sparsity = str(sparsity)
    execution_time_list = (execution_time_list)

    with open(latency_lookUpTable_dir, 'r') as latency_log:
        latency_log_content = latency_log.read()

    latency_dict = json.loads(latency_log_content) if latency_log_content != '' else {}
    latency_dict[tile_size] = latency_dict.get(tile_size, {})
    latency_dict[tile_size][sparsity] = execution_time_list

    with open(latency_lookUpTable_dir, 'w') as latency_log:
        latency_log.write(json.dumps(latency_dict))

def read_from_lookUpTable(tile_size:int, sparsity: int):
    tile_size = str(tile_size)
    sparsity = str(sparsity)
    with open(latency_lookUpTable_dir, 'r') as latency_log:
        latency_log_content = latency_log.read()

    latency_dict = json.loads(latency_log_content) if latency_log_content != '' else {}
    if tile_size in latency_dict and sparsity in latency_dict[tile_size]:
        return latency_dict[tile_size][sparsity][0]+latency_dict[tile_size][sparsity][1]
    return None
def get_execution_time(sparsity: int, tile_size: int, M, N, K, N_pruned_global, dtype, target_name:str):
    
    # read from latency look up table
    execution_time = read_from_lookUpTable(tile_size, sparsity)
    if execution_time != None:
        return execution_time

    K_pruned_max = math.floor((100-sparsity)/100 * K)
    block_num = (N_pruned_global + tile_size - 1)//tile_size
    N_ori_per_block = N // block_num
        

    # generate random mask_k for Ansor
    mask_k_test = np.zeros((block_num, K_pruned_max)).astype('int32')
    for row in range(block_num):
        mask_k_test[row, :] = np.random.choice(K, K_pruned_max, replace=False)
    mask_k_test.sort(axis=1)

    # create search taske
    if target_name in ['cuda', 'gpu']:
        target = tvm.target.cuda(arch='sm_70')
        device = tvm.device(target.get_target_device_type(), 0)
    else :
        target = tvm.target.Target(target="llvm", host="llvm")
        device = tvm.device(target.kind.name, 0)

    task = tvm.auto_scheduler.SearchTask(func=tw_kernel, args=(M, K, K_pruned_max, tile_size, block_num, dtype), target=target, 
                                        task_inputs={'mask_k': tvm.runtime.ndarray.array(mask_k_test)}, task_inputs_overwrite=True)

    # Set parameters for auto_scheduler
    log_file = os.path.join(root_dir, "Ansor/tw_{M}_{N}_{K}_ts{tile_size}_spar{sparsity}_{N_pruned_global}_{dtype}_{target_name}_trails{trails}.json".format(\
        M=M, N=N, K=K, tile_size=tile_size, sparsity=sparsity,N_pruned_global=N_pruned_global, dtype=dtype, target_name=target_name, trails=ansor_num_trials))
    if is_tune:
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=ansor_num_trials,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
            verbose=ansor_verbose,
        )
        cost_model = auto_scheduler.XGBModel()
        if is_resume:
            cost_model.update_from_file(log_file)

        if is_resume:
            search_policy = auto_scheduler.SketchPolicy(
                task,
                program_cost_model=cost_model,
                init_search_callbacks=[
                    auto_scheduler.PreloadMeasuredStates(log_file)
                ],
            )
        else:
            search_policy = auto_scheduler.SketchPolicy(
                task,
                program_cost_model=cost_model,
            )

        # Run auto-tuning (search)
        task.tune(tune_option, search_policy=search_policy)

    # Apply the best schedule
    sch_tw_mainbody, args_tw_mainbody = task.apply_best(log_file)

    # Check correctness and evaluate performance
    tw_mainbody = tvm.build(sch_tw_mainbody, args_tw_mainbody, target)
    tw_coowrite_C = insert_row_to_C_kernel_tir(M, N, tile_size, block_num, dtype, target_name) # Get the extern function `coowrite`

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

    # check results
    tvm.testing.assert_allclose(C_transposed_test.numpy().T, A_transposed_test.numpy().T @ B_transposed_pruned.T, 1e-1)

    # Evaluate execution time.
    tw_mainbody_evaluator = tw_mainbody.time_evaluator(tw_mainbody.entry_name, device, min_repeat_ms=500)
    tw_mainbody_execution_time = np.median(tw_mainbody_evaluator(C_transposed_skipped_test, A_transposed_test, B_transposed_packed_test, mask_k_test).results) * 1000
    
    tw_coowrite_C_evaluator = tw_coowrite_C.time_evaluator(tw_coowrite_C.entry_name, device, min_repeat_ms=500)
    tw_coowrite_C_execution_time = np.median(tw_coowrite_C_evaluator(C_transposed_test, C_transposed_skipped_test, mask_n_test).results) * 1000

    print("\tExecution time of tw_mainbody: %.3f ms" % (tw_mainbody_execution_time))
    print("\tExecution time of tw_coowrite_C: %.3f ms" % (tw_coowrite_C_execution_time))
    
    add_to_lookUpTable(tile_size, sparsity, [tw_mainbody_execution_time, tw_coowrite_C_execution_time])

    if is_tune==False:
        print('\nCUDA Code of tw_mainbody is\n')
        print(tw_mainbody.imported_modules[0].get_source())

    return tw_mainbody_execution_time+tw_coowrite_C_execution_time
    # print(tw_mainbody.imported_modules[0].get_source())


if __name__=='__main__':
    get_execution_time(75, 256, 1024, 1024, 1024, 1024, 'float32', 'gpu')
