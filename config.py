# Genetic Algorithm Parameters
POPULATION_SIZE = 50
NUM_GENERATIONS = 100
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7
ELITISM_SIZE = 5

# Neural Network Parameters
INPUT_SIZE = 84 * 84 * 3  # Game screen dimensions
HIDDEN_LAYERS = [64, 32]
OUTPUT_SIZE = 8  # Number of possible actions

# Game Parameters
GAME_NAME = "SpaceInvaders-v0"  # Default game
MAX_STEPS = 1000
RENDER_TRAINING = False
SAVE_BEST_MODELS = True

# Training Parameters
EVALUATION_EPISODES = 3  # Number of episodes to evaluate each individual
MAX_FITNESS = 10000  # Maximum fitness score to achieve

# Visualization Parameters
PLOT_INTERVAL = 5  # Plot every N generations
SAVE_VIDEO_INTERVAL = 20  # Save video of best individual every N generations

# File Paths
MODEL_SAVE_PATH = "models/"
VIDEO_SAVE_PATH = "videos/"
PLOT_SAVE_PATH = "plots/" 