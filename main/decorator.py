import time


def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        print(f"Calling function: {func.__name__}", end=' / ')
        print(f"Execution time: {execution_time} milliseconds")

        return result

    return wrapper
