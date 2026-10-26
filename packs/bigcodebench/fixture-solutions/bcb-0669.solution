import itertools
import math

def task_func(x):
    """
    Find the key pair in a dictionary, x, which has the highest sum of the cosine of each of its values.

    Parameters:
    - x (dict): The dictionary of key-value pairs.

    Returns:
    - tuple: The pair of keys with the highest sum of the cosine of their values.

    Requirements:
    - itertools
    - math

    Example:
    >>> task_func({'a': 1, 'b': 2, 'c': 3})
    ('a', 'b')
    ('a', 'b')
    >>> task_func({'a': 1, 'b': 2, 'c': 3, 'd': 4})
    ('a', 'b')
    ('a', 'b')
    """
    pairs = list(itertools.combinations(x.keys(), 2))
    max_pair = max(pairs, key=lambda pair: math.cos(x[pair[0]]) + math.cos(x[pair[1]]))
    print(max_pair)

    return max_pair
