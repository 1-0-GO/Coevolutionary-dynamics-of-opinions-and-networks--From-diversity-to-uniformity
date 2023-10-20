import os
import pickle

def cache_to_file(file_path):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Check if the cache file already exists
            if os.path.exists(file_path):
                # Load cached data from the file
                with open(file_path, 'rb') as file:
                    cache = pickle.load(file)
            else:
                # If the cache file doesn't exist, create an empty cache
                cache = {}

            # Check if the function result is already cached
            key = (args, frozenset(kwargs.items()))
            if key in cache:
                result = cache[key]
            else:
                # Call the decorated function and cache the result
                result = func(*args, **kwargs)
                cache[key] = result

            return result

        # Register a function to save the cache to the file when the program exits
        import atexit

        @atexit.register
        def save_cache():
            with open(file_path, 'wb') as file:
                pickle.dump(cache, file)

        return wrapper

    return decorator

# Usage example
cache_file = 'cache_data.pkl'