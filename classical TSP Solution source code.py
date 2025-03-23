from itertools import permutations
from functools import lru_cache

# Number of cities
N = 7
INF = float('INF')

# Adjacency matrix representation of the graph
graph = [
    [0, 12, 10, INF, INF, INF, 12],  # City 1
    [12, 0, 8, 12, INF, INF, INF],   # City 2
    [10, 8, 0, INF, 9, INF, INF],    # City 3
    [INF, 12, INF, 0, 11, 11, INF],  # City 4
    [INF, INF, 9, 11, 0, 6, 7],      # City 5
    [INF, INF, INF, 11, 6, 0, 9],    # City 6
    [12, INF, INF, INF, 7, 9, 0]     # City 7
]

#Dynamic Programming with bit masking (Held-Karp algorithm)
@lru_cache(None)
def tsp(mask, pos):
    if mask == (1 << N) - 1: #All cities visited
        return graph[pos][0] # Return to starting city
    
    min_cost = INF
    for city in range(N):
        if (mask & (1 << city )) == 0 and graph[pos][city] != INF: # If city not visited
            new_cost = graph[pos][city] + tsp(mask | (1 << city), city)
            min_cost = min(min_cost, new_cost)

    return min_cost

# Start TSP from City 1 (index 0)
print("Minimum cost using Dynamic Programming:", tsp(1,0))