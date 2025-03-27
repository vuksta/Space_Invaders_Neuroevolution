import tensorflow as tf
import numpy as np
from config import INPUT_SIZE, HIDDEN_LAYERS, OUTPUT_SIZE

class NeuralNetwork:
    def __init__(self, weights=None):
        self.model = self._build_model()
        if weights is not None:
            self.set_weights(weights)
    
    def _build_model(self):
        model = tf.keras.Sequential()
        
        # Input layer
        model.add(tf.keras.layers.Input(shape=(INPUT_SIZE,)))
        
        # Hidden layers
        for units in HIDDEN_LAYERS:
            model.add(tf.keras.layers.Dense(units, activation='relu'))
        
        # Output layer
        model.add(tf.keras.layers.Dense(OUTPUT_SIZE, activation='softmax'))
        
        return model
    
    def predict(self, state):
        return self.model.predict(state, verbose=0)
    
    def get_weights(self):
        weights = []
        for layer in self.model.layers:
            weights.extend(layer.get_weights())
        return weights
    
    def set_weights(self, weights):
        weight_index = 0
        for layer in self.model.layers:
            layer_weights = []
            for _ in range(len(layer.get_weights())):
                layer_weights.append(weights[weight_index])
                weight_index += 1
            layer.set_weights(layer_weights)
    
    def mutate(self, mutation_rate):
        weights = self.get_weights()
        for i in range(len(weights)):
            if np.random.random() < mutation_rate:
                mutation = np.random.normal(0, 0.1, weights[i].shape)
                weights[i] += mutation
        self.set_weights(weights)
    
    @staticmethod
    def crossover(parent1, parent2):
        weights1 = parent1.get_weights()
        weights2 = parent2.get_weights()
        child_weights = []
        
        for w1, w2 in zip(weights1, weights2):
            mask = np.random.random(w1.shape) < 0.5
            child_w = np.where(mask, w1, w2)
            child_weights.append(child_w)
        
        return NeuralNetwork(child_weights) 