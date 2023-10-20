import os
import joblib

import hashlib



def generate_cache_key(*args, **kwargs):
    # Create a cache key based on function arguments and keyword arguments
    k = list(kwargs.items())
    k.sort(key= lambda x: x[0])
    key = str((args, tuple(k)))
    return key
    return hashlib.md5(key.encode()) # this

def cached_function(cache_dir, cache_filename):
    def decorator(original_function):
        def wrapper(*args, **kwargs):
            cache_key = generate_cache_key(*args, **kwargs)
            cache_file = os.path.join(cache_dir, f'{cache_filename}_{cache_key}.joblib') # this sometimes returns different hashes for the same key because the order of the kwargs is not always the same, to fix this

            try:

                # Check if the cache file exists
                if os.path.exists(cache_file):
                    # Load cached results
                    print("loading from cache")
                    return joblib.load(cache_file)  # or pickle.load(open(cache_path, 'rb'))
                elif os.path.exists(
                        (os.path.expanduser("~/remote_cache/Coevolutionary-dynamics-of-opinions-and-networks--From-diversity-to-uniformity/"+ cache_file)
                        )
                ):

                    # Load cached results
                    print("loading from remote cache")
                    return joblib.load(
                        (os.path.expanduser("~/remote_cache/Coevolutionary-dynamics-of-opinions-and-networks--From-diversity-to-uniformity/"+ cache_file)
                         )
                    )
            # If cache doesn't exist, call the original function
            except:
                print("ERROR loading from cache")
                print(cache_file)


            print("calling original function")
            result = original_function(*args, **kwargs)

            # Save the result to the cache
            #joblib.dump(result, cache_path)  # or pickle.dump(result, open(cache_path, 'wb'))
            # Save the result with the arguments to the cache
            joblib.dump(result, os.path.expanduser("~/remote_cache/Coevolutionary-dynamics-of-opinions-and-networks--From-diversity-to-uniformity/"+ cache_file))  # or pickle.dump(result, open(cache_path, 'wb'))
            print("saved to cache")

            return result

        return wrapper
    print("initing cache...")

    return decorator

# a more efficient way to cache a function is
# to use the joblib library, which is a part of the scikit-learn library.
# It is a wrapper around the pickle library, which is used to serialize Python objects.
# The joblib library is more efficient than pickle because it can cache the results of a function
# in memory, which is faster than writing to disk.


