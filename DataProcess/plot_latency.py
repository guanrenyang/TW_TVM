import json
import matplotlib.pyplot as plt
import os
import math
def convert_keys_to_int(d):
    new_dict = {}
    for key, value in d.items():
        if isinstance(value, dict):
            new_dict[int(key)] = convert_keys_to_int(value)
        else:
            new_dict[int(key)] = value
    return new_dict

with open('/home/ryguan/TileSparsity_TVM/AutoSchedule/Outputs/latency_lookup_table.json', 'r') as f:
    data = convert_keys_to_int(json.load(f))
    

#.................granularity fixed.............................


picturePath = './pictures/'
if os.path.exists(picturePath)==False:
    os.mkdir(picturePath)

for granularity, spar2lat in data.items():
    x = spar2lat.keys()
    
    y_pack_and_gemm = []
    y_scatter = []
    y_total = []
    for lat_list in spar2lat.values():
        y_pack_and_gemm.append(lat_list[0])
        y_scatter.append(lat_list[1])
        y_total.append(lat_list[0]+lat_list[1])

    plt.plot(x, y_pack_and_gemm, label='pack then gemm')
    plt.plot(x, y_scatter, label='scatter')
    plt.plot(x, y_total, label='total', )


    plt.xlabel('sparsity')
    plt.ylabel('latency (ms)')
    plt.title('graularity={}'.format(granularity))
    # Add horizontal line
    # plt.axhline(y=0.22, color='red')

    # Save the figure
    plt.savefig(os.path.join(picturePath, 'latency_with_granularity={}_fixed.png'.format(granularity)))

    plt.clf()

sparsity_list = [i for i in range(50, 99, 5)]
for target_sparsity in sparsity_list:
    granularity_list = []
    y_pack_and_gemm = []
    y_scatter = []
    y_total = []
    for granularity, spar2lat in data.items():
        for sparsity, [pack_and_gemm, scatter] in spar2lat.items():
            if(sparsity==target_sparsity):
                granularity_list.append(granularity)
                y_pack_and_gemm.append(pack_and_gemm)
                y_scatter.append(scatter)
                y_total.append(pack_and_gemm+scatter)
    
    plt.plot(granularity_list, y_pack_and_gemm, label='pack then gemm')
    plt.plot(granularity_list, y_scatter, label='scatter')
    plt.plot(granularity_list, y_total, label='total')


    plt.xlabel('granularity')
    plt.ylabel('latency (ms)')
    plt.title('sparsity={}'.format(target_sparsity/100))
    # Add horizontal line
    # plt.axhline(y=0.22, color='red')

    # Save the figure
    plt.savefig(os.path.join(picturePath, 'latency_with_sparsity={}_fixed.png'.format(target_sparsity/100)))

    plt.clf()
    # ....................plot log2 of granularity..............
    granularity_list = [math.log2(i) for i in granularity_list]
    plt.plot(granularity_list, y_pack_and_gemm, label='pack then gemm')
    plt.plot(granularity_list, y_scatter, label='scatter')
    plt.plot(granularity_list, y_total, label='total')


    plt.xlabel('log2 granularity')
    plt.ylabel('latency (ms)')
    plt.title('sparsity={}'.format(target_sparsity/100))
    # Add horizontal line
    # plt.axhline(y=0.22, color='red')

    # Save the figure
    plt.savefig(os.path.join(picturePath, 'latency_with_sparsity={}_fixed_granularity_log2.png'.format(target_sparsity/100)))

    plt.clf()
