# Space Invaders Neuroevolution Engine

A genetic algorithm that evolves neural networks to master Space Invaders through natural selection, featuring tournament selection, mutation, and 50 competing strategies per generation. The system processes raw game pixels through TensorFlow networks, with the most fascinating outcome being emergent, unprogrammed strategies that develop purely through evolutionary pressure.

![Space Invaders AI](https://i.imgur.com/example.gif)

## Features

- **Neural Networks**: TensorFlow-based networks (84x84x3 → [64,32] → 6 actions)
- **Genetic Algorithm**: 50 networks/generation, tournament selection, 0.1 mutation rate
- **Evolution**: Elitism preservation, adaptive mutation, crossover operations
- **Training**: 100 generations, parallel evaluation, 1000+ point performance
- **Visualization**: Real-time fitness tracking, strategy recording, evolution graphs

## Quick Start

```bash
# Clone and install
git clone https://github.com/vuksta/Space_Invaders_Neuroevolution.git
cd Space_Invaders_Neuroevolution
pip install -r requirements.txt

# Import ROM
python -m retro.import /path/to/SpaceInvaders.rom

# Start evolution
python train.py
```

## System Architecture

```
raw pixels → preprocessing → neural networks → actions → reward → selection → reproduction → mutation
```

- **Input**: 84x84x3 normalized pixel values
- **Network**: Hidden layers [64,32], ReLU activation
- **Output**: 6 actions (NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE)
- **Selection**: Tournament style, top performers reproduce
- **Evolution**: Elite preservation, 0.1 mutation rate, 0.7 crossover

## Results

- **Performance**: 1000+ points by generation 60
- **Strategies**: Emergent patterns like rhythmic shooting, position cycling
- **Efficiency**: Learns without human demonstration or explicit programming
- **Convergence**: Clear fitness improvements across generations

## Implementation

```python
# Example: neural network prediction and action
action_probs = neural_network.predict(state.reshape(1, -1))[0]
action = np.argmax(action_probs)
next_state, reward, done, info = env.step(action)

# Example: mutation operation
def mutate(self, mutation_rate=0.1):
    weights = self.get_weights()
    for i in range(len(weights)):
        if np.random.random() < mutation_rate:
            mutation = np.random.normal(0, 0.1, weights[i].shape)
            weights[i] += mutation
    self.set_weights(weights)
```

## References

- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*
- Stanley, K. O., & Miikkulainen, R. (2002). Evolving Neural Networks through Augmenting Topologies
- Such, F. P., et al. (2017). Deep Neuroevolution: Genetic Algorithms Are a Competitive Alternative for Training Deep Neural Networks for Reinforcement Learning