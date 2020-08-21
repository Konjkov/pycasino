from psutil import cpu_count
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

num_proc = cpu_count(logical=False)


def pool(function):

    def wrapper(*args):
        with ProcessPoolExecutor() as executor:
            async_result = [executor.submit(function, *args) for _ in range(num_proc)]
        return np.array([res.result() for res in async_result])

    return wrapper


def thread(function):

    def wrapper(*args):
        with ThreadPoolExecutor() as executor:
            async_result = [executor.submit(function, *args) for _ in range(num_proc)]
        return np.array([res.result() for res in async_result])

    return wrapper
