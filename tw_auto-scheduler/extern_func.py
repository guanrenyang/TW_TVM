import tvm
from tvm import te

def insert_row_to_C_kernel(M, N, tile_size, block_num, dtype='float32'):
    C_transposed_skipped = te.placeholder((block_num, tile_size, M), name='C_transposed_skipped', dtype=dtype)
    mask_n = te.placeholder((block_num, tile_size), name='mask_n', dtype='int32') 
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

        # with irb.for_range(0, N, kind='serial', name='n') as n:
        #     with irb.for_range(0, M, kind='serial', name='m') as m:
        #         out_ptr[n * M + m] = tvm.tir.generic.cast(0, data.dtype)

        with irb.for_range(0, block_num, kind='parallel', name='bn') as bn:
            with irb.for_range(0, tile_size, kind='parallel', name='ts') as ts:
                with irb.for_range(0, M, kind='parallel', name='col') as col:
                    '''Now, pointer in TensorIR can be accessed as a un-flattened tensor'''
                    with irb.new_scope():
                        out_ptr[(N_ori_per_block * bn + mask_n_ptr[bn, ts]), col] += data_ptr[bn, ts, col]
        sch = irb.get()
        print(type(sch))
        return schs
        
    C_transposed = te.extern((N, M),
                            [C_transposed_skipped, mask_n],
                            lambda ins, outs: write_C_to_sparse(ins[0], ins[1], outs[0]),
                            dtype=dtype,
                            name='C_transposed',
                            )
    s = te.create_schedule(C_transposed.op)
    
    # xo, yo, xi, yi = s[C_transposed].tile(C_transposed.op.axis[0], C_transposed.op.axis[1], x_factor=32, y_factor=32)
    
    # s[C_transposed].bind(xo, te.thread_axis('blockIdx.x'))
    # s[C_transposed].bind(yo, te.thread_axis('blockIdx.y'))
    # s[C_transposed].bind(xi, te.thread_axis('threadIdx.x'))
    # s[C_transposed].bind(yi, te.thread_axis('threadIdx.y'))

    return s, [C_transposed, C_transposed_skipped, mask_n]

sch, args = insert_row_to_C_kernel(1024, 1024, 128, 8, 'float32')
# print(tvm.lower(sch, args, simple_mode=True).script()) 