from ..AutoSchedule.tw_auto_scheduler import get_execution_time
import math
def tw_grid_search(M, N, K, N_pruned_global, dtype, target_name, search_space: list):

    log_file = "tw_{M}_{N}_{K}_{K_pruned_max}_{N_pruned_global}_{dtype}_{target_name}".format(M=M, N=N, K=K, K_pruned_max=K_pruned_max, N_pruned_global=N_pruned_global, dtype=dtype, target_name=target_name)
    with open(log_file, 'w') as output_file:
        print("Settings:",
            "M=%d" %(M),
            "N=%d" %(N),
            "K=%d" %(K),
            "N_pruned_global=%d" %(N_pruned_global),
            "dtye=%s" %(dtype),
            "target=%s" %target_name,
            file=output_file
        )
        
        # results_dict = {}

        min_execution_time = float('inf')
        min_tile_size = 0 
        for tile_size in search_space['tile_size']:
            for sparsity in search_space['sparsity']:
                
                K_pruned_max = math.floor((1-sparsity) * K)
                print('Try tile_size=%d, sparsity=%.3f on %s'%(tile_size, sparsity, target_name), file=output_file)

                # Get Latency
                execution_time_mainbody, execution_time_coowrite = get_execution_time(M, N, K, tile_size, K_pruned_max, N_pruned_global, dtype, target_name)
                execution_time = execution_time_mainbody+execution_time_coowrite
                
                # Get Accuracy
                
                # results_dict[tile_size] = execution_time
                print('Execution time is %.3f+%.3f=%.3f ms' % (execution_time_mainbody,execution_time_coowrite,execution_time), file=output_file)
                if(execution_time<min_execution_time):
                    min_execution_time = execution_time
                    min_tile_size = tile_size
            print("The best tile_size in %s is %d"%(target_name, min_tile_size), file=output_file)
            print("The best execution time is %.3f\n" % (min_execution_time), file=output_file)
            # print("All results on %s is "%target_name, results_dict, file=output_file)
            print("\n", file=output_file)

def grid_search():
    
    search_space = {"tile_size":[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],\
                    'sparsity' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
    dtype = 'float32'

    # tw_kernel = register_tw_kernel(1024, 1024, 1024, 512, 1024, dtype, 'gpu', search_space)
    
    tw_grid_search(1024, 1024, 1024, 512, 1024, dtype, 'gpu', search_space)
    tw_grid_search(1024, 1024, 1024, 512, 1024, dtype, 'cpu', search_space)

if __name__=='__main__':
    grid_search()
