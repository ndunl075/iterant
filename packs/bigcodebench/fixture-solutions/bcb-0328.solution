import collections
import random
from queue import PriorityQueue


def task_func(number_teams=5):
    """
    Create a random sports ranking and sort it by points in descending order.
    
    Note:
    - Each team is assigned a name in the format "Team i" and a corresponding random number of points, where i ranges from 1 to the specified number of teams. 
    - The ranking is then sorted in descending order of points and returned as an OrderedDict.

    Parameters:
    number_teams (int, optional): The number of teams in the ranking. Default is 5.

    Returns:
    OrderedDict: Sorted dictionary where keys are team names and values are points.

    Requirements:
    - collections
    - random
    - queue.PriorityQueue


    Example:
    >>> random.seed(0)
    >>> ranking = task_func()
    >>> print(ranking)
    OrderedDict([('Team 4', 50), ('Team 5', 40), ('Team 1', 30), ('Team 2', 20), ('Team 3', 10)])
    """

    # Constants
    
    TEAMS = []
    POINTS = []

    for i in range(1, number_teams+1):
        TEAMS.append("Team "+str(i))
        POINTS.append(10*i)
    
    shuffled_points = POINTS.copy()
    random.shuffle(shuffled_points)
    ranking = dict(zip(TEAMS, shuffled_points))

    sorted_ranking = PriorityQueue()
    for team, points in ranking.items():
        sorted_ranking.put((-points, team))

    sorted_ranking_dict = collections.OrderedDict()
    while not sorted_ranking.empty():
        points, team = sorted_ranking.get()
        sorted_ranking_dict[team] = -points

    return sorted_ranking_dict
