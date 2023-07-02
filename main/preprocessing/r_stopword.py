from decorator import *


@timer_decorator
def compute_sum(numbers):
    return sum(numbers)


numbers = [1, 2, 3, 4, 5]
result = compute_sum(numbers)
print(result)
