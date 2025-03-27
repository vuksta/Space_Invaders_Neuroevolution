# Evolutionary Game AI

This project implements a neuroevolutionary approach to game AI, where neural networks evolve through genetic algorithms to master classic video games. The system maintains a population of neural networks, each representing a different playing strategy, and uses evolutionary operators (selection, crossover, and mutation) to iteratively improve their performance. Through this process, the networks learn optimal game-playing behaviors by evolving their weights and architecture.

## 🎮 Features

- **Neural Network Architecture**
  - Customizable network topology
  - TensorFlow-based implementation
  - Support for various input/output configurations
  - Weight mutation and crossover operations

- **Genetic Algorithm Implementation**
  - Tournament selection
  - Elitism preservation
  - Adaptive mutation rates
  - Uniform and multi-point crossover
  - Population diversity maintenance

- **Game Environment**
  - Integration with gym-retro
  - State preprocessing pipeline
  - Reward shaping capabilities
  - Episode recording and visualization

- **Training Pipeline**
  - Parallel fitness evaluation
  - Progress tracking and visualization
  - Model checkpointing
  - Performance metrics logging

## 🛠️ Technical Stack

- Python 3.9+
- TensorFlow 2.13.0
- gym-retro 0.8.0
- NumPy 1.24.3
- OpenCV 4.8.0
- NEAT-Python 0.92

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/evolutionary-game-ai.git
cd evolutionary-game-ai
```

2. Create and activate a conda environment:
```bash
conda create -n evolutionary_game python=3.9
conda activate evolutionary_game
```

3. Install dependencies:
```bash
conda install numpy=1.24.3 tensorflow=2.13.0 opencv
pip install gym-retro==0.8.0 neat-python==0.92
```

4. Import ROMs:
```bash
python -m retro.import /path/to/your/roms
```

## 🎯 Usage

### Basic Training

```python
from game_environment import GameEnvironment
from genetic_algorithm import GeneticAlgorithm

# Initialize environment
env = GameEnvironment()

# Create genetic algorithm instance
ga = GeneticAlgorithm(env)

# Start training
ga.train()
```

### Custom Configuration

Modify `config.py` to adjust parameters:

```python
# Genetic Algorithm Parameters
POPULATION_SIZE = 50
NUM_GENERATIONS = 100
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7
ELITISM_SIZE = 5

# Neural Network Architecture
INPUT_SIZE = 84 * 84 * 3
HIDDEN_LAYERS = [64, 32]
OUTPUT_SIZE = 8
```

## 🔧 Architecture

### Neural Network (`neural_network.py`)
- Implements feed-forward neural networks
- Supports weight mutation and crossover
- Handles model serialization

### Genetic Algorithm (`genetic_algorithm.py`)
- Population management
- Selection mechanisms
- Evolution operators
- Fitness evaluation

### Game Environment (`game_environment.py`)
- ROM integration
- State preprocessing
- Action execution
- Reward calculation

### Visualization (`visualization.py`)
- Training progress plots
- Performance metrics
- Video recording
- Generation statistics

## 📊 Performance Metrics

- Average fitness per generation
- Best fitness achieved
- Population diversity
- Training convergence rate
- Episode completion rate

## 🎯 Supported Games

- Space Invaders
- Donkey Kong
- Mario Bros
- Other gym-retro compatible games

## 🔍 Implementation Details

### State Preprocessing
```python
def preprocess_state(self, state):
    # Resize to 84x84
    state = cv2.resize(state, (84, 84))
    # Normalize pixel values
    state = state / 255.0
    return state.flatten()
```

### Neural Network Architecture
```python
def _build_model(self):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(INPUT_SIZE,)),
        *[tf.keras.layers.Dense(units, activation='relu') 
          for units in HIDDEN_LAYERS],
        tf.keras.layers.Dense(OUTPUT_SIZE, activation='softmax')
    ])
    return model
```

### Genetic Operators
```python
def mutate(self, mutation_rate):
    weights = self.get_weights()
    for i in range(len(weights)):
        if np.random.random() < mutation_rate:
            mutation = np.random.normal(0, 0.1, weights[i].shape)
            weights[i] += mutation
    self.set_weights(weights)
```

## 📈 Results

The AI typically achieves:
- Space Invaders: 1000+ points
- Donkey Kong: Level 2 completion
- Mario Bros: World 1-1 completion

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.