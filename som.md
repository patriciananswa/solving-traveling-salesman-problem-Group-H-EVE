1. Input: Distance matrix of cities
2. Convert distance matrix into 2D coordinates using Multi-Dimensional Scaling (MDS)

3. Initialize SOM parameters:
   - num_neurons = Number of cities (one neuron per city)
   - num_iterations = 1000
   - initial_learning_rate = 0.8
   - initial_radius = num_neurons / 2

4. Initialize neurons in a circular formation

5. Define helper functions:
   - EuclideanDistance(a, b): Compute Euclidean distance between two points
   - FindClosestNeuron(city, neurons): Find the nearest neuron to a given city

6. Train SOM over 'num_iterations':
   a. Decay learning rate and neighborhood radius over time
   b. Shuffle cities randomly
   c. For each city in shuffled list:
      i.   Find the closest neuron (winner)
      ii.  Update winner and neighboring neurons using learning rule

7. Assign each city to the closest neuron
8. Sort neurons by their position to form the final route
9. Ensure the route starts and ends at City 1
10. Calculate total route distance using the original distance matrix
11. Output:
   - Final optimized route
   - Total distance
   - Visualization of the final route
