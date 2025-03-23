import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

# Define the distance matrix for the 7 cities (replace with your graph data)
distance_matrix = np.array([
    [0, 10, 15, 20, 25, 30, 35],  # City 1
    [10, 0, 12, 18, 22, 28, 32],   # City 2
    [15, 12, 0, 10, 14, 20, 25],   # City 3
    [20, 18, 10, 0, 8, 15, 20],    # City 4
    [25, 22, 14, 8, 0, 10, 12],     # City 5
    [30, 28, 20, 15, 10, 0, 8],     # City 6
    [35, 32, 25, 20, 12, 8, 0]      # City 7
])

# Convert distance matrix to 2D coordinates using MDS
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
cities = mds.fit_transform(distance_matrix)

# SOM parameters
num_neurons = 7  # One neuron per city
num_iterations = 1000  # Number of training iterations
initial_learning_rate = 0.8  # Initial learning rate
initial_radius = num_neurons / 2  # Initial neighborhood radius

# Initialize neurons in a circle around the center of the cities
center = np.mean(cities, axis=0)
theta = np.linspace(0, 2 * np.pi, num_neurons, endpoint=False)
neurons = np.array([center + [np.cos(t), np.sin(t)] for t in theta])

# Function to calculate Euclidean distance between two points
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

# Function to find the closest neuron (winner) for a given city
def find_closest_neuron(city, neurons):
    distances = [euclidean_distance(city, neuron) for neuron in neurons]
    return np.argmin(distances)

# Function to extract the final route from the neurons
def extract_route(neurons, cities):
    route = []
    for city in cities:
        closest_neuron = find_closest_neuron(city, neurons)
        route.append(closest_neuron)
    return route

# Training loop
for iteration in range(num_iterations):
    # Decay learning rate and neighborhood radius
    learning_rate = initial_learning_rate * (1 - iteration / num_iterations)
    radius = initial_radius * (1 - iteration / num_iterations)

    # Shuffle cities for each iteration
    shuffled_cities = cities[np.random.permutation(len(cities))]

    # For each city, find the closest neuron (winner) and update neurons
    for city in shuffled_cities:
        winner = find_closest_neuron(city, neurons)

        # Update the winner and its neighbors
        for i, neuron in enumerate(neurons):
            distance_to_winner = abs(i - winner)
            if distance_to_winner <= radius:
                # Update neuron position
                neurons[i] += learning_rate * (city - neurons[i])

# Extract the final route
final_route = extract_route(neurons, cities)

# Ensure the route starts and ends at City 1
final_route = [0] + final_route + [0]  # City 1 is index 0

# Calculate the total distance of the final route
total_distance = 0
for i in range(len(final_route) - 1):
    total_distance += distance_matrix[final_route[i], final_route[i + 1]]

# Output the results
print("Final Route:", final_route)
print("Total Distance:", total_distance)

# Visualize the final route
plt.scatter(cities[:, 0], cities[:, 1], c='blue', label='Cities')
plt.scatter(neurons[:, 0], neurons[:, 1], c='red', label='Neurons')
plt.plot(cities[final_route, 0], cities[final_route, 1], linestyle='-', marker='o', label='Route')
plt.title("SOM-TSP Final Route")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.legend()
plt.show()