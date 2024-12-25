import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
target_image = cv2.imread(r'C:\Users\Dell\Desktop\fruit1.jpg')
# Define parameters
population_size = 1000
mutation_rate = 0.1 # Initial mutation rate 0.1
max_generations = 20 # Limited to 20 iterations
num_parents = int(population_size * 0.2)
# Define image dimensions
image_height, image_width, _ = target_image.shape
# Initialize the population with some individuals resembling the target image
def initialize_population():
  population = [np.clip(target_image.copy() + np.random.randint(-30, 30, (image_height,
  image_width, 3), dtype=np.int16), 0, 255).astype(np.uint8) for _ in range(population_size)]
  return population
# Fitness function (MSE between individual's image and target image)
def fitness(individual):
  mse = np.mean((individual - target_image) ** 2)
  return mse
# Plotting the fitness development
def plot_fitness(fitness_values):
  plt.plot(fitness_values)
  plt.title('Fitness Development Over Generations')
  plt.xlabel('Generation')
  plt.ylabel('Fitness')
  plt.show()
# Visualize and save an individual as an image file
def visualize_individual(individual, output_filename):
  cv2.imwrite(output_filename, individual)
  plt.imshow(cv2.cvtColor(individual, cv2.COLOR_BGR2RGB)) # Display image in JupyterNotebook
  plt.show()
# Selection
def selection(population, num_parents):
  fitness_values = [fitness(individual) for individual in population]
  total_fitness = sum(fitness_values)
  probabilities = [fit / total_fitness for fit in fitness_values]
  selected_parents_indices = np.random.choice(len(population), num_parents, p=probabilities)
  selected_parents = [population[i] for i in selected_parents_indices]
  return selected_parents
# Crossover (Recombination): Blend Images
def crossover(parent1, parent2):
  alpha = random.uniform(0.3, 0.7) # Blend factor
  child = cv2.addWeighted(parent1, alpha, parent2, 1 - alpha, 0)
  return child
# Mutation: Random Pixel Mutation
def mutate(individual, mutation_rate):
  if random.random() < mutation_rate:
    i, j, k = random.randint(0, image_height - 1), random.randint(0, image_width - 1),
    random.randint(0, 2)
    individual[i, j, k] = random.randint(0, 255)
  return individual
# Genetic algorithm loop with selection, crossover, and mutation
def genetic_algorithm(mutation_rate):
  fitness_values=[]
  population = initialize_population()
  for generation in range(max_generations):
    population = sorted(population, key=fitness)
    best_individual = population[0]
    best_fitness = fitness(best_individual)
    print(f"Generation {generation}: Best Fitness = {best_fitness}")
    fitness_values.append(best_fitness) # Store fitness value for plotting
    if best_fitness < 1.0: # Early stopping if fitness is close to 0 (exact match)
      break
    # Play the images continuously and pause 0.5 second before each image
    plt.ion()
    plt.pause(0.5)
    # Visualize and save the best individual
    output_filename = f"generation_{generation}.png"
    visualize_individual(best_individual, output_filename)
    # Selection
    num_parents = int(population_size * 0.2)
    parents = selection(population, num_parents)
    # Crossover
    offspring = []
    for i in range(0, len(parents), 2):
      if i + 1 < len(parents):
        child = crossover(parents[i], parents[i + 1])
        offspring.append(child)
    # Mutation
    if generation < max_generations / 2:
      mutation_rate *= 0.9 # Gradually reduce mutation rate
    offspring = [mutate(individual, mutation_rate) for individual in offspring]
    # Replace old population with new population
    population = offspring + [best_individual]
  # Plotting the fitness development
  plot_fitness(fitness_values)
# Run the genetic algorithm
genetic_algorithm(mutation_rate)
