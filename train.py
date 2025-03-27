import os
import numpy as np
from game_environment import GameEnvironment
from genetic_algorithm import GeneticAlgorithm
from visualization import (
    plot_fitness_history, plot_generation_stats,
    create_training_summary, save_training_video
)
from config import (
    PLOT_INTERVAL, SAVE_VIDEO_INTERVAL,
    PLOT_SAVE_PATH, VIDEO_SAVE_PATH, MODEL_SAVE_PATH
)

def create_directories():
    """Create necessary directories for saving results."""
    directories = [PLOT_SAVE_PATH, VIDEO_SAVE_PATH, MODEL_SAVE_PATH]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main():
    # Create necessary directories
    create_directories()
    
    # Initialize game environment
    game_env = GameEnvironment()
    
    try:
        # Initialize genetic algorithm
        ga = GeneticAlgorithm(game_env)
        
        # Train the population
        ga.train()
        
        # Plot final results
        plot_fitness_history(
            ga.fitness_history,
            save_path=f"{PLOT_SAVE_PATH}fitness_history.png"
        )
        
        # Save final best model video
        if ga.best_individual is not None:
            save_training_video(
                game_env,
                ga.best_individual,
                "final"
            )
        
        # Print training summary
        print(create_training_summary(
            ga.fitness_history,
            ga.best_fitness,
            len(ga.fitness_history)
        ))
        
    finally:
        # Clean up
        game_env.close()

if __name__ == "__main__":
    main() 