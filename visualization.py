import matplotlib.pyplot as plt
import numpy as np
from config import PLOT_SAVE_PATH, VIDEO_SAVE_PATH

def plot_fitness_history(fitness_history, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history)
    plt.title('Training Progress')
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_generation_stats(fitness_scores, generation, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.hist(fitness_scores, bins=20)
    plt.title(f'Fitness Distribution - Generation {generation}')
    plt.xlabel('Fitness')
    plt.ylabel('Count')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def create_training_summary(fitness_history, best_fitness, num_generations):
    summary = f"""
    Training Summary:
    ----------------
    Total Generations: {num_generations}
    Best Fitness Achieved: {best_fitness:.2f}
    Final Average Fitness: {fitness_history[-1]:.2f}
    Improvement: {((fitness_history[-1] - fitness_history[0]) / fitness_history[0] * 100):.2f}%
    """
    return summary

def save_training_video(game_env, neural_network, generation):
    video_path = f"{VIDEO_SAVE_PATH}generation_{generation}.mp4"
    game_env.record_episode(neural_network, video_path)
    return video_path 