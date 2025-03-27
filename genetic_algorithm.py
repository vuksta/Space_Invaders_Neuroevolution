import numpy as np
from neural_network import NeuralNetwork
from config import (
    POPULATION_SIZE, NUM_GENERATIONS, MUTATION_RATE,
    CROSSOVER_RATE, ELITISM_SIZE, EVALUATION_EPISODES
)

class GeneticAlgorithm:
    """
    The main evolutionary algorithm that:
    - Maintains a population of neural networks
    - Evaluates their performance
    - Creates new generations through selection, crossover, and mutation
    - Keeps track of the best performers
    """
    
    def __init__(self, game_env):
        """
        Initialize the genetic algorithm with:
        - A fresh population of neural networks
        - The game environment for evaluation
        - Tracking variables for best performers
        """
        self.game_env = game_env
        self.population = [NeuralNetwork() for _ in range(POPULATION_SIZE)]
        self.best_fitness = float('-inf')
        self.best_individual = None
        self.fitness_history = []
    
    def evaluate_population(self):
        """
        Test each neural network in the population.
        Returns fitness scores and updates best performers.
        """
        fitness_scores = []
        for individual in self.population:
            fitness = self.game_env.evaluate_individual(individual, EVALUATION_EPISODES)
            fitness_scores.append(fitness)
            
            # Update best individual if we found a better one
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_individual = individual
        
        self.fitness_history.append(np.mean(fitness_scores))
        return fitness_scores
    
    def select_parents(self, fitness_scores):
        """
        Choose parents for the next generation using tournament selection.
        This is like a competition where the best performers get to "reproduce".
        """
        tournament_size = 3
        parents = []
        
        for _ in range(POPULATION_SIZE):
            # Randomly select a few individuals for the tournament
            tournament = np.random.choice(
                POPULATION_SIZE, tournament_size, replace=False
            )
            tournament_fitness = [fitness_scores[i] for i in tournament]
            winner = tournament[np.argmax(tournament_fitness)]
            parents.append(self.population[winner])
        
        return parents
    
    def create_next_generation(self, parents):
        """
        Create a new generation of neural networks through:
        - Elitism (keeping the best performers)
        - Crossover (combining successful strategies)
        - Mutation (adding random changes)
        """
        new_population = []
        
        # Keep the best performers (elitism)
        sorted_indices = np.argsort([self.game_env.evaluate_individual(ind) for ind in self.population])
        for i in range(ELITISM_SIZE):
            new_population.append(self.population[sorted_indices[-i-1]])
        
        # Create children through crossover and mutation
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = np.random.choice(parents, 2, replace=False)
            
            if np.random.random() < CROSSOVER_RATE:
                # Create child by combining parents
                child = NeuralNetwork.crossover(parent1, parent2)
            else:
                # Create fresh child
                child = NeuralNetwork()
            
            # Add some random changes
            child.mutate(MUTATION_RATE)
            new_population.append(child)
        
        self.population = new_population
    
    def train(self):
        """
        Main training loop that:
        1. Evaluates current population
        2. Selects parents
        3. Creates next generation
        4. Tracks progress
        5. Saves best models
        """
        for generation in range(NUM_GENERATIONS):
            print(f"Generation {generation + 1}/{NUM_GENERATIONS}")
            
            # Evaluate current population
            fitness_scores = self.evaluate_population()
            
            # Select parents
            parents = self.select_parents(fitness_scores)
            
            # Create next generation
            self.create_next_generation(parents)
            
            # Print statistics
            print(f"Best Fitness: {self.best_fitness:.2f}")
            print(f"Average Fitness: {np.mean(fitness_scores):.2f}")
            
            # Save best model if needed
            if self.best_individual is not None:
                self.save_best_model(generation)
    
    def save_best_model(self, generation):
        """
        Save the best performing neural network.
        Useful for later use or analysis.
        """
        if self.best_individual is not None:
            weights = self.best_individual.get_weights()
            np.save(f"models/best_model_gen_{generation}.npy", weights) 