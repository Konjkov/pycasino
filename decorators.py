from multiprocessing import Pool, Process, cpu_count

import numpy as np


def multi_process(function):
    """https://www.ellicium.com/python-multiprocessing-pool-process/"""

    def wrapper(*args):
        num_proc = cpu_count() // 2
        pool = Pool(num_proc)
        async_result = [pool.apply_async(function, args) for i in range(num_proc)]
        pool.close()
        pool.join()
        return np.array([res.get() for res in async_result])

    return wrapper
