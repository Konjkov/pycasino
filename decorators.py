from psutil import cpu_count
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

num_proc = cpu_count(logical=False)


def pool(function):
    """https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor

    The ProcessPoolExecutor class is an Executor subclass that uses a pool of processes
    to execute calls asynchronously. ProcessPoolExecutor uses the multiprocessing module,
    which allows it to side-step the Global Interpreter Lock but also means that only
    picklable objects can be executed and returned.
    """

    def wrapper(*args):
        with ProcessPoolExecutor(max_workers=num_proc) as executor:
            async_result = [executor.submit(function, *args) for _ in range(num_proc)]
            return np.array([res.result() for res in async_result])

    return wrapper


def thread(function):
    """https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor

    ThreadPoolExecutor is an Executor subclass that uses a pool of threads to execute
    calls asynchronously.
    """

    def wrapper(*args):
        with ThreadPoolExecutor(max_workers=num_proc) as executor:
            async_result = [executor.submit(function, *args) for _ in range(num_proc)]
            return np.array([res.result() for res in async_result])

    return wrapper
