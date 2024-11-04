import numpy as np

# Objective function (can be modified for different problems)
def objective_function(position):
    # Example: Rastrigin function (a common multimodal benchmark function)
    return 10 * len(position) + sum([x**2 - 10 * np.cos(2 * np.pi * x) for x in position])

# GSO Parameters
num_glowworms = 30       # Number of glowworms
dimension = 2            # Problem dimensionality
lower_bound = -5.12      # Lower bound of the search space
upper_bound = 5.12       # Upper bound of the search space
iterations = 100         # Number of iterations
luciferin_decay = 0.4    # Decay rate of luciferin
luciferin_enhancement = 0.6 # Enhancement of luciferin after each iteration
neighborhood_radius = 0.5 # Initial neighborhood radius
step_size = 0.03         # Movement step size

# Initialize glowworm positions and luciferin levels
glowworms = np.random.uniform(lower_bound, upper_bound, (num_glowworms, dimension))
luciferin = np.zeros(num_glowworms)

# Main GSO loop
for iteration in range(iterations):
    # Update luciferin levels based on objective function values
    for i in range(num_glowworms):
        luciferin[i] = (1 - luciferin_decay) * luciferin[i] + luciferin_enhancement * (1 / (1 + objective_function(glowworms[i])))

    # Move glowworms towards neighbors with higher luciferin
    for i in range(num_glowworms):
        # Find neighbors within the neighborhood radius with higher luciferin levels
        neighbors = []
        for j in range(num_glowworms):
            if i != j:
                distance = np.linalg.norm(glowworms[i] - glowworms[j])
                if distance < neighborhood_radius and luciferin[j] > luciferin[i]:
                    neighbors.append(j)

        # Move towards a randomly selected neighbor with higher luciferin
        if neighbors:
            chosen_neighbor = glowworms[np.random.choice(neighbors)]
            direction = (chosen_neighbor - glowworms[i])
            direction /= np.linalg.norm(direction)  # Normalize direction
            glowworms[i] += step_size * direction  # Move towards neighbor

    # Optional: Update neighborhood radius (dynamic radius adjustment)
    neighborhood_radius = max(0.1, neighborhood_radius * 0.95)  # Reduce radius gradually

    # Output best solution found in this iteration
    best_idx = np.argmin([objective_function(g) for g in glowworms])
    print(f"Iteration {iteration + 1}, Best Position: {glowworms[best_idx]}, Best Value: {objective_function(glowworms[best_idx])}")

# Final best position and value
best_idx = np.argmin([objective_function(g) for g in glowworms])
print("\nOptimal Position:", glowworms[best_idx])
print("Optimal Value:", objective_function(glowworms[best_idx]))