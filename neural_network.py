import tensorflow as tf
import numpy as np
from config import INPUT_SIZE, HIDDEN_LAYERS, OUTPUT_SIZE

class NeuralNetwork:
    """
    A neural network that can evolve through genetic operations.
    Think of this as the "brain" of our game-playing AI.
    """
    
    def __init__(self, weights=None):
        """
        Create a new neural network.
        If weights are provided, use them to initialize the network.
        Otherwise, create a fresh network with random weights.
        """
        self.model = self._build_model()
        if weights is not None:
            self.set_weights(weights)
    
    def _build_model(self):
        """
        Build the neural network architecture.
        We're using a simple feed-forward network with:
        - Input layer: Takes the game screen (84x84x3 pixels)
        - Hidden layers: Two layers of neurons (64 and 32)
        - Output layer: 8 possible game actions
        """
        model = tf.keras.Sequential()
        
        # Input layer - takes flattened game screen
        model.add(tf.keras.layers.Input(shape=(INPUT_SIZE,)))
        
        # Hidden layers - these are where the magic happens!
        for units in HIDDEN_LAYERS:
            model.add(tf.keras.layers.Dense(units, activation='relu'))
        
        # Output layer - decides what action to take
        model.add(tf.keras.layers.Dense(OUTPUT_SIZE, activation='softmax'))
        
        return model
    
    def predict(self, state):
        """
        Make a decision based on the current game state.
        Returns probabilities for each possible action.
        """
        return self.model.predict(state, verbose=0)
    
    def get_weights(self):
        """
        Get all the weights from the network.
        Used for saving/loading and genetic operations.
        """
        weights = []
        for layer in self.model.layers:
            weights.extend(layer.get_weights())
        return weights
    
    def set_weights(self, weights):
        """
        Set all the weights in the network.
        Used for loading saved networks and applying genetic operations.
        """
        weight_index = 0
        for layer in self.model.layers:
            layer_weights = []
            for _ in range(len(layer.get_weights())):
                layer_weights.append(weights[weight_index])
                weight_index += 1
            layer.set_weights(layer_weights)
    
    def mutate(self, mutation_rate):
        """
        Randomly change some weights in the network.
        This is like the "mutation" in biological evolution.
        Higher mutation_rate = more random changes.
        """
        weights = self.get_weights()
        for i in range(len(weights)):
            if np.random.random() < mutation_rate:
                # Add some random noise to the weights
                mutation = np.random.normal(0, 0.1, weights[i].shape)
                weights[i] += mutation
        self.set_weights(weights)
    
    @staticmethod
    def crossover(parent1, parent2):
        """
        Create a new network by combining two parent networks.
        This is like biological reproduction - the child inherits
        traits from both parents.
        """
        weights1 = parent1.get_weights()
        weights2 = parent2.get_weights()
        child_weights = []
        
        # Randomly mix weights from both parents
        for w1, w2 in zip(weights1, weights2):
            mask = np.random.random(w1.shape) < 0.5
            child_w = np.where(mask, w1, w2)
            child_weights.append(child_w)
        
        return NeuralNetwork(child_weights) 