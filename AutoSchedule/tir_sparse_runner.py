import logging
from contextlib import contextmanager
from typing import Any,Callable, List, Optional, Union
import numpy as np
import scipy as spy
import tvm
from tvm.meta_schedule.runner.local_runner import LocalRunner
from tvm.meta_schedule.runner.utils import (
    T_ARG_INFO_JSON_OBJ_LIST,
    T_ARGUMENT_LIST,
    alloc_argument_common,
    run_evaluator_common,
)
from tvm.meta_schedule.utils import derived_object, get_global_func_with_default_on_worker
from tvm.runtime import Device,Module,ndarray
T_ARG_INFO_JSON_OBJ = List[Any] 



def sparse_fixed_runner(block_num: int, tile_size: int, N: int):

    def sparse_fixed_alloc_argument_cuda(
        device:Device,
        args_info: T_ARG_INFO_JSON_OBJ_LIST,
        alloc_repeat: int,
    ) -> List[T_ARGUMENT_LIST]: 
        f_random_fill = get_global_func_with_default_on_worker(
            name="tvm.contrib.random.random_fill_for_measure", default=None
        )
        def alloc_tensor(_, dtype, shape) -> ndarray.NDArray:
            arg = ndarray.empty(shape=shape, dtype=dtype, device=device)
            f_random_fill(arg)
            return arg
        def generate_indices(block_num, tile_size, N):
            # generate mask_n
            mask_n_test = np.zeros((block_num, tile_size)).astype('int32')
            for row in range(block_num):
                mask_n_test[row, :] = np.random.choice(N//block_num, tile_size, replace=False)
            mask_n_test.sort(axis=1)

            # indices=np.array(indices).astype('int32')
            arg=ndarray.array(mask_n_test,device=device)
            return arg

        repeated_args: List[T_ARGUMENT_LIST]=[]
        for _ in range(alloc_repeat):
            args:T_ARGUMENT_LIST=[]
            args_info:T_ARG_INFO_JSON_OBJ
        
            for arg_info in args_info:
                if arg_info[1]=='float32' or arg_info[1]=='float16':
                    #matrix data
                    arg:Any=alloc_tensor(*arg_info)
                elif arg_info[1]=='int32':
                    #indices
                    arg:Any=generate_indices(block_num, tile_size, N)
                else:
                    raise NotImplementedError(arg_info)
                args.append(arg)
            repeated_args.append(args)
        return repeated_args
    
    return LocalRunner(f_alloc_argument=sparse_fixed_alloc_argument_cuda)