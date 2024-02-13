from src.test import some_function

try:
    some_function()
except ValueError as e:
    print(f"Caught an exception: {e}")