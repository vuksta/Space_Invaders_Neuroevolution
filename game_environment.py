import gym
import numpy as np
import cv2
from config import GAME_NAME, MAX_STEPS

class GameEnvironment:
    def __init__(self):
        self.env = gym.make(GAME_NAME, render_mode='rgb_array')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    
    def preprocess_state(self, state):
        # Resize and normalize the state
        state = cv2.resize(state, (84, 84))
        state = state / 255.0
        return state.flatten()
    
    def evaluate_individual(self, neural_network, num_episodes=1):
        total_reward = 0
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            state = self.preprocess_state(state)
            done = False
            steps = 0
            
            while not done and steps < MAX_STEPS:
                # Get action from neural network
                action_probs = neural_network.predict(state.reshape(1, -1))[0]
                action = np.argmax(action_probs)
                
                # Take action in environment
                state, reward, done, _, _ = self.env.step(action)
                state = self.preprocess_state(state)
                
                total_reward += reward
                steps += 1
        
        return total_reward / num_episodes
    
    def record_episode(self, neural_network, output_path):
        state, _ = self.env.reset()
        done = False
        frames = []
        
        while not done:
            frames.append(self.env.render())
            state = self.preprocess_state(state)
            action_probs = neural_network.predict(state.reshape(1, -1))[0]
            action = np.argmax(action_probs)
            state, _, done, _, _ = self.env.step(action)
        
        # Save frames as video
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
        
        for frame in frames:
            out.write(frame)
        out.release()
    
    def close(self):
        self.env.close() 