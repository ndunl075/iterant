import random
import string

def task_func(max_length, n_samples, seed=None):
    """Generate a list containing random strings of lowercase letters. Each string's length varies from 1 to `max_length`.
    An optional seed can be set for the random number generator for reproducible results.

    Note:
    The function utilizes the `random.choices` function to generate random strings and combines them into a list.

    Parameters:
    max_length (int): The maximum length of the strings.
    n_samples (int): The number of strings to return.
    seed (int, optional): A seed for the random number generator. If None, the generator is initialized without a seed.

    Returns:
    list: A list containing random strings. Each string is a random combination of lowercase letters, 
    and their lengths will vary from 1 to `max_length`.

    Requirements:
    - random
    - string

    Raises:
    ValueError: If max_length is smaller than 1.

    Example:
    >>> task_func(3, 12, seed=12)
    ['gn', 'da', 'mq', 'rp', 'aqz', 'ex', 'o', 'b', 'vru', 'a', 'v', 'ncz']
    >>> task_func(5, n_samples=8, seed=1)
    ['ou', 'g', 'tmjf', 'avlt', 's', 'sfy', 'aao', 'rzsn']

    """
    # Handling negative input
    if max_length < 1:
        raise ValueError("max_length must be larger than or equal to 1.")

    # Constants within the function for better encapsulation
    LETTERS = string.ascii_lowercase

    # Setting the seed for the random number generator for reproducibility
    if seed is not None:
        random.seed(seed)

    all_combinations = []

    for i in range(n_samples):
        random_length = random.randint(1, max_length)
        combination = ''.join(random.choices(LETTERS, k=random_length))
        all_combinations.append(combination)


    # Simplifying the reduction using native functionality
    return all_combinations
