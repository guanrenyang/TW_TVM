import numpy as np


def tiled_wise_1d(sparsity, tile_size, weight, metric='l2norm'):
    
    assert(len(weight.shape)==2)

    block_num = weight.shape[1] // tile_size
    if weight.shape[1] % tile_size == 0:
        split_weight = np.split(weight, block_num, axis=1)
    else:
        split_weight = []
        if block_num != 0:
            split_weight = np.split(weight[:,0:block_num * tile_size], block_num, axis=1)
        split_weight.append(weight[:,block_num * tile_size:weight.shape[1]])
    
    split_l2_norm = [ np.linalg.norm(w, axis=1) / w.shape[1] for w in split_weight ]
    split_l2_norm_flatten = [ v for n in split_l2_norm for v in n]
    threshold = np.percentile(split_l2_norm_flatten, sparsity)
    
    #New mask
    split_mask = [ norm > threshold for norm in split_l2_norm]
    split_mask = [ mask.reshape(mask.shape[0], 1) for mask in split_mask ]
    split_mask = [ np.tile(m, (1, split_weight[i].shape[1])).reshape(split_weight[i].shape) for i, m in enumerate(split_mask) ]
    mask = np.concatenate(split_mask, axis = 1).astype('int')

    assert(mask.shape == weight.shape)

    return np.linalg.norm(mask * weight)

pruning_algos = {
    # "ew"  : element_wise,
    # "vw"  : vector_vise,
    # "bw"  : block_wise,
    "tw1" : tiled_wise_1d,
    # "tw2" : tiled_wise_2d,
    # "tw1m" : tiled_wise_1d_mix,
    # "tw2m" : tiled_wise_2d_mix,
    # "twvw" : tw_vw_4choose2,
    # "twvw16" : tw_vw_16choosen,
}

def get_metric(pruning_type):
    return pruning_algos[pruning_type]

if __name__ == '__main__':
    weight = np.random.rand(1024, 1024)
