from AutoSchedule.tw_auto_scheduler import  get_execution_time
from Pruning.layer_pruner import get_metric
import numpy as np
import math, os, json, time

cublas_execution_time = 0.22

np.random.seed(42)

root_dir = '/home/ryguan/TileSparsity_TVM/Outputs/'
if not os.path.exists(root_dir):
    os.mkdir(root_dir)

# search_result_dir = os.path.join(root_dir, 'search_results.json')
search_log_dir = os.path.join(root_dir, 'search_log_'+time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())))

# def add_to_result(tile_size, sparsity, latency_metric: dict, accuracy_metric: dict):
#     with open(search_result_dir, 'w+') as search_result_file:
#         result_dict = json.loads(search_result_file.read()) if search_result_file!='' else {}
        
#         result_dict[tile_size] = result_dict.get(tile_size, {})
#         result_dict[sparsity] = result_dict[tile_size].get(sparsity, {})
        
#         result_dict[tile_size][sparsity].update(latency_metric)
#         result_dict[tile_size][sparsity].update(accuracy_metric)
        
#         search_result_file.write(json.dumps(result_dict))


def add_to_log(obj):
    with open(search_log_dir, 'a') as log_file:
        print(obj, file=log_file)

def object_function(arruracy_metric, latency_metric): # to minimize
    return arruracy_metric + latency_metric * latency_metric 

def tw_grid_search(weight, pruning_type, M, N, K, N_pruned_global, dtype, target_name, search_space: list):
 
    add_to_log("Settings: grid_{pruneType}_{M}_{N}_{K}_{dtype}_{target_name}".format(pruneType=pruning_type, M=M, N=N, K=K, dtype=dtype, target_name=target_name))

    best_obj = float('inf') 
    best_tile_size = 0 
    best_sparsity = 0
    best_execution_time = 0
    best_l2_norm_ratio = 0
    
    search_task_id = 1
    total_search_tasks = len(search_space['tile_size']) * len(search_space['sparsity'])
    for tile_size in search_space['tile_size']:
        for sparsity in search_space['sparsity']:

            start_time = time.time()
            # Get Accuracy
            l2_norm = get_metric(pruning_type)(sparsity=sparsity, tile_size=tile_size, weight=weight)
            l2_norm_ratio = float(np.linalg.norm(weight)/l2_norm) # greater than 1
            
            # Get Latency
            execution_time = get_execution_time(sparsity, tile_size, M, N, K, N_pruned_global, dtype, target_name)
            execution_time_ratio = execution_time / cublas_execution_time

            obj = object_function(l2_norm_ratio ,execution_time_ratio)
            
            add_to_log('Search Task[%d/%d] Time %.3fs Tile_size %d, Sparsity %d, Platform %s, ExecutionTime %.3f, l2NormRatio %.3f'\
                       %(search_task_id, total_search_tasks, time.time() -
                           start_time, tile_size, sparsity, target_name, execution_time, l2_norm_ratio))
            search_task_id += 1
            
            if(obj<best_obj):
                best_obj = obj
                best_tile_size = tile_size
                best_sparsity = sparsity
                best_execution_time = execution_time
                best_l2_norm_ratio = l2_norm_ratio

            if(execution_time<=cublas_execution_time):
                add_to_log('\t Find an efficient Kernel, tile_size={tile_size}, sparsity={sparsity}, exec_time={exec_time}'.format(tile_size=tile_size, sparsity=sparsity, exec_time=execution_time))
            
    add_to_log("Final Results: Tile_size {}, Sparsity {}, Platform {}, ExecutionTime {:.3f}, l2NormRatio {:.3f}".format(best_tile_size, best_sparsity, target_name, best_execution_time, best_l2_norm_ratio))


def grid_search(weight, input, pruneType):
    
    assert(weight.shape[0]==input.shape[1])
    M, K, N = input.shape[0], input.shape[1], weight.shape[1]

    search_space = {"tile_size":[2, 4, 8, 16, 32, 64, 128, 256, 512],\
                    'sparsity' : [i for i in range(50, 100, 5)]}

    dtype = 'float16' if weight.dtype==np.float16 else 'float32'
    # Now only tw-1-dim is inplementted
    if pruneType=='tw1':
        tw_grid_search(weight, 'tw1', M, N, K, N, dtype, 'gpu', search_space)
    


if __name__=='__main__':
    dtype = 'float32'
    weight = np.random.rand(1024, 1024).astype(dtype)
    input = np.random.rand(1024, 1024).astype(dtype)
    pruneType = 'tw1'
    grid_search(weight, input, pruneType)
