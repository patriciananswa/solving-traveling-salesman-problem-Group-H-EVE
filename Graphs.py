# Representation of the graph using an adjacency matrix
# 0 indicates no direct path between cities
graph = [                      #The graph is represented using a 0-based index, meaning:
    [0, 12, 10, 0, 0, 0, 12],  # City 1 → Index 0
    [12, 0, 8, 12, 0, 0, 0],   # City 2 → Index 1
    [10, 8, 0, 11, 3, 0, 9],   # City 3 → Index 2
    [0, 12, 11, 0, 11, 10, 0], # City 4 → Index 3
    [0, 0, 3, 11, 0, 6, 7],    # City 5 → Index 4
    [0, 0, 0, 10, 6, 0, 9],    # City 6 → Index 5
    [12, 0, 9, 0, 7, 9, 0]     # City 7 → Index 6
]

print(graph[0][1])






