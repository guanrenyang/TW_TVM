import tvm
from tvm.ir.module import IRModule
from tvm.script import ir as I
from tvm.script import tir as T
import numpy as np
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

        with irb.for_range(0, block_num, kind='serial', name='bn') as bn:
            with irb.for_range(0, tile_size, kind='serial', name='ts') as ts:
                with irb.for_range(0, M, kind='serial', name='col') as col:
                    '''Now, pointer in TensorIR can be accessed as a un-flattened tensor'''
                    out_ptr[(N_ori_per_block * bn + mask_n_ptr[bn, ts]), col] += data_ptr[bn, ts, col]
        sch = irb.get()
        # print(sch)
        return sch
        
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



def insert_row_to_C_kernel_script(M, N, tile_size, block_num, dtype='float32'):
    N_ori_per_block = N // block_num
    @I.ir_module
    class Module:
        @T.prim_func
        def main(C_transposed_h: T.handle, C_transposed_skipped_h: T.handle, mask_n_h: T.handle):
            T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
            C_transposed = T.match_buffer(C_transposed_h, (N, M), dtype)
            C_transposed_skipped = T.match_buffer(C_transposed_skipped_h, (block_num, tile_size, M), dtype)
            mask_n = T.match_buffer(mask_n_h, (block_num, tile_size), 'int32')
            for bn in range(block_num):
                for ts in range(tile_size):
                    for col in range(M):
                        with T.block('B'):
                                C_transposed[(N_ori_per_block * bn + mask_n[bn, ts]), col] += C_transposed_skipped[bn, ts, col]

    ir_module = Module
    sch = tvm.tir.Schedule(ir_module)
    
    block_b = sch.get_block('B')
    (bn, ts, m) = sch.get_loops(block_b)
    m_o, m_i = sch.split(m, factors=[M//32, 32])
    
    sch.bind(bn, 'blockIdx.z')
    sch.bind(ts, 'threadIdx.y')
    sch.bind(m_o, 'blockIdx.x')
    sch.bind(m_i, 'threadIdx.x')

    print(ir_module.script())
insert_row_to_C_kernel_script(2048, 1024, 128, 8, 'float32')
# sch, args = insert_row_to_C_kernel(2048, 1024, 128, 8, 'float32')
# ir_module = tvm.lower(sch, args, simple_mode=True)
# sch = tvm.tir.Schedule(ir_module)


# print(ir_module.script()) 