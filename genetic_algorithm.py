import numpy as np
from neural_network import NeuralNetwork
from config import (
    POPULATION_SIZE, NUM_GENERATIONS, MUTATION_RATE,
    CROSSOVER_RATE, ELITISM_SIZE, EVALUATION_EPISODES
)

class GeneticAlgorithm:
    def __init__(self, game_env):
        self.game_env = game_env
        self.population = [NeuralNetwork() for _ in range(POPULATION_SIZE)]
        self.best_fitness = float('-inf')
        self.best_individual = None
        self.fitness_history = []
    
    def evaluate_population(self):
        fitness_scores = []
        for individual in self.population:
            fitness = self.game_env.evaluate_individual(individual, EVALUATION_EPISODES)
            fitness_scores.append(fitness)
            
            # Update best individual
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_individual = individual
        
        self.fitness_history.append(np.mean(fitness_scores))
        return fitness_scores
    
    def select_parents(self, fitness_scores):
        # Tournament selection
        tournament_size = 3
        parents = []
        
        for _ in range(POPULATION_SIZE):
            tournament = np.random.choice(
                POPULATION_SIZE, tournament_size, replace=False
            )
            tournament_fitness = [fitness_scores[i] for i in tournament]
            winner = tournament[np.argmax(tournament_fitness)]
            parents.append(self.population[winner])
        
        return parents
    
    def create_next_generation(self, parents):
        new_population = []
        
        # Elitism
        sorted_indices = np.argsort([self.game_env.evaluate_individual(ind) for ind in self.population])
        for i in range(ELITISM_SIZE):
            new_population.append(self.population[sorted_indices[-i-1]])
        
        # Create children through crossover and mutation
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = np.random.choice(parents, 2, replace=False)
            
            if np.random.random() < CROSSOVER_RATE:
                child = NeuralNetwork.crossover(parent1, parent2)
            else:
                child = NeuralNetwork()
            
            child.mutate(MUTATION_RATE)
            new_population.append(child)
        
        self.population = new_population
    
    def train(self):
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
        if self.best_individual is not None:
            weights = self.best_individual.get_weights()
            np.save(f"models/best_model_gen_{generation}.npy", weights) 