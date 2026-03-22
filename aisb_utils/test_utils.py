from termcolor import colored
from typing import Callable
from functools import wraps

def red(text):
    return colored(text, "red")

def green(text):
    return colored(text, "green")

def report(test_func: Callable) -> Callable:
    name = f"{test_func.__module__}.{test_func.__name__}"

    @wraps(test_func)
    def wrapper(*args, **kwargs):
        try:
            out = test_func(*args, **kwargs)
        except Exception as e:
            print(red(f"{name} failed."))
            raise e
        else:
            print(green(f"{name} passed."))
            return out
    return wrapper
