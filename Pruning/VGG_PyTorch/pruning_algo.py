# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import pdb



def kernel2matrix(kernel):
    # Get the shape of the kernel
    out_channel, in_channel, kernel_height, kernel_width = kernel.shape
    
    # Reshape the kernel to (out_channel, in_channel * kernel_height * kernel_width)
    # kernel = kernel.reshape(out_channel, -1)
    
    # Create the transformation matrix
    transformation_matrix = np.zeros((out_channel * kernel_height * kernel_width, 
                                       in_channel * kernel_height * kernel_width))
    
    for i in range(out_channel):
        for j in range(in_channel):
            for k in range(kernel_height):
                for l in range(kernel_width):
                    index = (i * kernel_height * kernel_width) + (k * kernel_width) + l
                    transformation_matrix[index, j*(kernel_height*kernel_width)+(k*kernel_width)+l] = kernel[i,j,k,l]
    
    return transformation_matrix
def matrix2kernel(transformation_matrix, out_channel, in_channel, kernel_height, kernel_width):
    # Compute the size of the output kernel
    kernel = np.zeros((out_channel, in_channel, kernel_height, kernel_width))
    
    for i in range(out_channel):
        for j in range(in_channel):
            for k in range(kernel_height):
                for l in range(kernel_width):
                    index = (i * kernel_height * kernel_width) + (k * kernel_width) + l
                    kernel_index = (i, j, k, l)
                    kernel[kernel_index] = transformation_matrix[index, j*(kernel_height*kernel_width)+(k*kernel_width)+l]
    
    return kernel

def img2col_forward(accumulated_scores):
    new_accumulated_scores = []
    for l in accumulated_scores:
        if len(l.shape) == 4:
            layer_2d = np.reshape(l, (l.shape[0]*l.shape[1]*l.shape[2],l.shape[3]))
        if len(l.shape) == 2:
            layer_2d = l
        new_accumulated_scores.append(abs(layer_2d))
    return new_accumulated_scores

def img2col_back_ward(masks, mask_shape):
    new_masks = []
    for l in range(len(masks)):
        m = masks[l]
        s = mask_shape[l].shape
        if len(s) == 2:
            mask = m
        if len(s) == 4:
            mask = np.reshape(m, (s[0],s[1],s[2],s[3]))
        new_masks.append(mask)
    return new_masks

def element_wise(accumulated_scores, sparsity):
    # Element_wise
    # Sparsity
    scores = []
    for layer in accumulated_scores:
        scores = scores + layer.flatten().tolist()
    scores = np.array(scores)
    threshold = np.percentile(abs(scores), sparsity)
    
    #New mask
    new_mask_values = []
    for layer in range(len(accumulated_scores)):
        mask = np.array(abs(accumulated_scores[layer]) > threshold)
        new_mask_values.append(mask)
    
    return new_mask_values

def tiled_wise_2d(accumulated_scores, sparsity, granularity = 64):
    # prune N + K dimension on whole network
    # N dimension pruning whole column
    # K dimension pruning uses granularity pruning
    # prune N dimesnion first is necessary, and hence consider the pruning on K and N independently. 
    
    sparsity = 100 - np.sqrt(10000 - sparsity * 100)

    # pruning on N
    l2_norm_list = [ np.linalg.norm(score, axis=0) / score.shape[0] for score in accumulated_scores ]
    split_layer_l2_norm_list = [ v for n in l2_norm_list for v in n]
    threshold = np.percentile(split_layer_l2_norm_list, sparsity)
    mask = [ norm > threshold for norm in l2_norm_list ]
    tiled_mask = [ np.tile(m, (score.shape[0], 1)).reshape(score.shape) for m, score in zip(mask, accumulated_scores)]
    pruned_accumulated_scores = [ m * score for m, score in zip(tiled_mask, accumulated_scores) ]
    
    # pruning on K
    l2_norm_list = []
    split_score_list = []
    for layer in range(len(pruned_accumulated_scores)):
        score = pruned_accumulated_scores[layer]
        split_point = []
        column_number = 0
        column_number_list = []
        for column_index in range(score.shape[1]):
            if mask[layer][column_index] == 1:
                column_number += 1
            if column_number == granularity:
                if column_index + 1 < score.shape[1]:
                    split_point.append(column_index + 1)
                    column_number_list.append(column_number)
                    column_number = 0
        column_number_list.append(column_number)
        split_score = np.split(score, split_point, axis=1)
        split_score_list.append(split_score)
        split_l2_norm = [ np.linalg.norm(w, axis=1) / w.shape[1] for i, w in enumerate(split_score) ]
        l2_norm_list.append(split_l2_norm)

    split_layer_l2_norm_list = [ v for n in l2_norm_list for v in n]
    split_vector_l2_norm_list = [ v for n in split_layer_l2_norm_list for v in n]
    threshold = np.percentile(split_vector_l2_norm_list, sparsity)

    #New mask
    new_mask_values = []
    for layer in range(len(pruned_accumulated_scores)):
        score = pruned_accumulated_scores[layer]
        split_score = split_score_list[layer]
        split_mask = [ norm > threshold for norm in l2_norm_list[layer] ]
        split_mask = [ mask.reshape(mask.shape[0], 1) for mask in split_mask ]
        split_mask = [ np.tile(m, (1, split_score[i].shape[1])).reshape(split_score[i].shape) for i, m in enumerate(split_mask) ]
        mask = np.concatenate(split_mask, axis = 1)
        
        assert(mask.shape == tiled_mask[layer].shape)
        assert(tiled_mask[layer].shape == accumulated_scores[layer].shape)
        new_mask_values.append(mask * tiled_mask[layer])

    return new_mask_values

def tiled_wise_1d(accumulated_scores, sparsity, granularity = 64):
    # granularity pruning on whole layer (K dimension for a K*N matrix)
    l2_norm_list = []
    split_score_list = []
    for layer in range(len(accumulated_scores)):
        score = accumulated_scores[layer]

        block_num = score.shape[1] // granularity
        if score.shape[1] % granularity == 0:
            split_score = np.split(score, block_num, axis=1)
        else:
            split_score = []
            if block_num != 0:
                split_score = np.split(score[:,0:block_num * granularity], block_num, axis=1)
            split_score.append(score[:,block_num * granularity:score.shape[1]])
        
        split_l2_norm = [ np.linalg.norm(w, axis=1) / w.shape[1] for w in split_score ]
        l2_norm_list.append( )
        split_score_list.append(split_score)
        
    
    split_layer_l2_norm_list = [ v for n in l2_norm_list for v in n]

    split_weight_l2_norm_list = [ v for n in split_layer_l2_norm_list for v in n]
    threshold = np.percentile(split_weight_l2_norm_list, sparsity)
    
    #New mask
    new_mask_values = []
    for layer in range(len(accumulated_scores)):
        split_score = split_score_list[layer]

        split_mask = [ norm > threshold for norm in l2_norm_list[layer] ]
        split_mask = [ mask.reshape(mask.shape[0], 1) for mask in split_mask ]
        split_mask = [ np.tile(m, (1, split_score[i].shape[1])).reshape(split_score[i].shape) for i, m in enumerate(split_mask) ]

        mask = np.concatenate(split_mask, axis = 1)
        assert(mask.shape == accumulated_scores[layer].shape)
        new_mask_values.append(mask)

    return new_mask_values


def tw_mix_fn(accumulated_scores, g1percent, layer_mask):
    # For G = 1
    l2_norm_list = []
    for layer in range(len(accumulated_scores)):
        score = accumulated_scores[layer]
        block_num = score.shape[1]
        split_score = np.split(score, block_num, axis=1)
        split_l2_norm = [ np.linalg.norm(w, axis=1) / w.shape[1] for w in split_score ]
        mask = layer_mask[layer]
        split_l2_norm = split_l2_norm - (mask.transpose())
        l2_norm_list.append(split_l2_norm)
    
    split_layer_l2_norm_list = [ v for n in l2_norm_list for v in n]
    split_weight_l2_norm_list = [ v for n in split_layer_l2_norm_list for v in n]

    threshold = np.percentile(split_weight_l2_norm_list, 100 - g1percent)
    
    #New mask
    new_mask_values = []
    for layer in range(len(accumulated_scores)):
        split_mask = [ norm > threshold for norm in l2_norm_list[layer] ]
        split_mask = [ np.concatenate([np.expand_dims(m, axis=1) for i in range(1)], axis=1) for m in split_mask ]
        mask1 = np.concatenate(split_mask, axis = 1)
        mask0 = layer_mask[layer]
        mask = mask0 + mask1
        assert(mask.shape == accumulated_scores[layer].shape)
        new_mask_values.append(mask)

    return new_mask_values


def tiled_wise_1d_mix(accumulated_scores, sparsity, granularity = 64, g1percent = 5):

    layer_mask = tiled_wise_1d(accumulated_scores, sparsity + g1percent, granularity)

    return tw_mix_fn(accumulated_scores, g1percent, layer_mask)


def tiled_wise_2d_mix(accumulated_scores, sparsity, granularity = 64, g1percent = 1):
    
    layer_mask = tiled_wise_2d(accumulated_scores, sparsity + g1percent, granularity)

    return tw_mix_fn(accumulated_scores, g1percent, layer_mask)
    
    
def vector_vise(accumulated_scores, sparsity):
    new_mask_values = []
    for layer in range(len(accumulated_scores)):
        ## Vector_wise need transpose !
        score = np.transpose(accumulated_scores[layer])

        block_size = 16
        # topk = 4
        # assert(sparsity == int(100 * (1 - topk / block_size)))
        topk = int((1 - sparsity / 100) * block_size)
        mask = np.zeros(score.shape)
        for i in range(score.shape[0]):
            j = 0
            while j < score.shape[1]:
                j_end = j + block_size
                ## If index beyond the score.shape[1]
                if j_end > score.shape[1]:
                    j_end = score.shape[1]

                threshold = 0
                if topk + 1 < j_end - j:
                    threshold = np.sort(score[i][j:j_end])[-(topk+1)]

                mask[i][j:j_end] = score[i][j:j_end] > threshold
                j += block_size
                
        assert(score.shape == mask.shape)
        
        ## Vector_wise mask need reverse transpose !
        new_mask_values.append(np.transpose(mask))

    return new_mask_values

def block_wise(accumulated_scores, sparsity, granularity = 32):
    # block wise pruning on whole network
    l2_norm_list = []
    new_mask_values = []
    for layer in range(len(accumulated_scores)):
        score = accumulated_scores[layer]
        block_size = granularity
        i = 0
        while i < score.shape[0]:
            j = 0
            while j < score.shape[1]:
                l2_norm_value = np.linalg.norm(score[i:i+block_size, j:j+block_size])
                l2_norm_list.append(l2_norm_value)
                j += block_size
            i += block_size
    
    threshold = np.percentile(l2_norm_list, sparsity)
    block_mask = l2_norm_list > threshold
    mask_index = 0
    for layer in range(len(accumulated_scores)):
        score = accumulated_scores[layer]
        mask = np.zeros(score.shape)
        i = 0
        while i < mask.shape[0]:
            j = 0
            while j < mask.shape[1]:
                if block_mask[mask_index] == True:
                    mask[i:i+block_size, j:j+block_size] = 1
                mask_index += 1
                j += block_size
            i += block_size

        assert(accumulated_scores[layer].shape == mask.shape)
        new_mask_values.append(mask)

    return new_mask_values

def tw_vw_4choose2(accumulated_scores,sparsity,granularity = 64):
    # pdb.set_trace()
    sparsity = 100 - 2 * (100 - sparsity)
    block_size = 4
    topk = 2   #要剪topk个,但因为取整了，误差累积后可能影响较大
    
    if sparsity < 0:
        sparsity = 0
        topk = 3   #要剪topk个,但因为取整了，误差累积后可能影响较大

    #先计算出tw和vw的稀疏度，调用已有的tw函数
    tiled_mask = tiled_wise_1d(accumulated_scores,sparsity,granularity)
    pruned_accumulated_scores = [ m * score for m, score in zip(tiled_mask, accumulated_scores) ]
    
    new_mask_values = []
    for layer in range(len(pruned_accumulated_scores)):
        ## Vector_wise need transpose !
        score = np.transpose(pruned_accumulated_scores[layer])
        t_mask = np.transpose(tiled_mask[layer])

        mask = np.zeros(score.shape)
        for i in range(score.shape[0]):
            j = 0
            column_number = 0
            for column_index in range(score.shape[1]):
                if t_mask[i][column_index] == True: #没有在TW中被剪枝掉
                    column_number += 1   
                if column_number == block_size: #够block_size个元素了，对该向量剪枝
                    threshold = 0
                    # if topk + 1 < column_index - j:
                    threshold = np.sort(score[i][j:column_index+1])[-(topk+1)] 
                    mask[i][j:column_index+1] = score[i][j:column_index+1] > threshold 
                    j = column_index+1  #下一个vector的起始位置为j
                    column_number = 0
            if(column_number > 0):  #若vector长度不够block_size，用同样的思路只不过是少剪一点
                threshold = 0
                
                if topk < column_number:                    
                    threshold = np.sort(score[i][j:column_index+1])[-(topk+1)]

                mask[i][j:column_index+1] = score[i][j:column_index+1] > threshold           
            assert(score.shape == mask.shape)
        
        ## Vector_wise mask need reverse transpose !
        new_mask_values.append(np.transpose(mask * t_mask))  

    return new_mask_values

def tw_vw_16choosen(accumulated_scores,sparsity,granularity = 64):
    # pdb.set_trace()
    sparsity = 100 - np.sqrt(10000 - sparsity * 100)
    block_size = 16
    topk = int((1 - sparsity / 100) * block_size)   #要剪topk个,但因为取整了，误差累积后可能影响较大

    #先计算出tw和vw的稀疏度，调用已有的tw函数
    tiled_mask = tiled_wise_1d(accumulated_scores,sparsity,granularity)
    pruned_accumulated_scores = [ m * score for m, score in zip(tiled_mask, accumulated_scores) ]
    
    new_mask_values = []
    for layer in range(len(pruned_accumulated_scores)):
        ## Vector_wise need transpose !
        score = np.transpose(pruned_accumulated_scores[layer])
        t_mask = np.transpose(tiled_mask[layer])

        mask = np.zeros(score.shape)
        for i in range(score.shape[0]):
            j = 0
            column_number = 0
            for column_index in range(score.shape[1]):
                if t_mask[i][column_index] == True: #没有在TW中被剪枝掉
                    column_number += 1   
                if column_number == block_size: #够block_size个元素了，对该向量剪枝
                    threshold = 0
                    # if topk + 1 < column_index - j:
                    threshold = np.sort(score[i][j:column_index+1])[-(topk+1)] 
                    mask[i][j:column_index+1] = score[i][j:column_index+1] > threshold 
                    j = column_index+1  #下一个vector的起始位置为j
                    column_number = 0
            if(column_number > 0):  #若vector长度不够block_size，用同样的思路只不过是少剪一点
                threshold = 0
                
                if topk < column_number:                    
                    threshold = np.sort(score[i][j:column_index+1])[-(topk+1)]

                mask[i][j:column_index+1] = score[i][j:column_index+1] > threshold           
            assert(score.shape == mask.shape)
        
        ## Vector_wise mask need reverse transpose !
        new_mask_values.append(np.transpose(mask * t_mask))  

    return new_mask_values

pruning_algos = {
    "ew"  : element_wise,
    "vw"  : vector_vise,
    "bw"  : block_wise,
    "tw1" : tiled_wise_1d,
    "tw2" : tiled_wise_2d,
    "tw1m" : tiled_wise_1d_mix,
    "tw2m" : tiled_wise_2d_mix,
    "twvw" : tw_vw_4choose2,
    "twvw16" : tw_vw_16choosen,
}

def pruning_fun(pruning_type):
    return pruning_algos[pruning_type]
