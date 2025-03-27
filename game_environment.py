import gym
import numpy as np
import cv2
from config import (
    GAME_NAME, MAX_STEPS, SCORE_MULTIPLIER,
    SURVIVAL_BONUS, SHOT_PENALTY
)

class SpaceInvadersEnvironment:
    """
    Wrapper for the Space Invaders game environment that handles:
    - Game state preprocessing
    - Action execution
    - Reward calculation
    - Episode recording
    """
    
    def __init__(self):
        """
        Initialize the Space Invaders environment.
        We're using gym-retro which provides access to the classic game.
        """
        self.env = gym.make(GAME_NAME, render_mode='rgb_array')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.last_score = 0
        self.last_lives = 3
        self.last_shots = 0
    
    def preprocess_state(self, state):
        """
        Convert the raw Space Invaders screen into a format our neural network can understand.
        - Resize to 84x84 pixels (standard size for many RL papers)
        - Normalize pixel values to [0,1] range
        - Flatten the image for the neural network
        """
        # Resize and normalize the state
        state = cv2.resize(state, (84, 84))
        state = state / 255.0
        return state.flatten()
    
    def calculate_reward(self, info):
        """
        Calculate reward based on Space Invaders specific metrics:
        - Score changes
        - Lives remaining
        - Shots fired
        """
        reward = 0
        
        # Score-based reward
        if info.get('score', 0) > self.last_score:
            reward += (info['score'] - self.last_score) * SCORE_MULTIPLIER
        
        # Survival bonus
        if info.get('lives', 3) > self.last_lives:
            reward += SURVIVAL_BONUS
        
        # Shot penalty (to encourage efficient shooting)
        if info.get('shots', 0) > self.last_shots:
            reward -= SHOT_PENALTY
        
        # Update last values
        self.last_score = info.get('score', 0)
        self.last_lives = info.get('lives', 3)
        self.last_shots = info.get('shots', 0)
        
        return reward
    
    def evaluate_individual(self, neural_network, num_episodes=1):
        """
        Test how well a neural network plays Space Invaders.
        Returns the average reward across episodes.
        Higher reward = better performance.
        """
        total_reward = 0
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            state = self.preprocess_state(state)
            done = False
            steps = 0
            
            # Reset tracking variables
            self.last_score = 0
            self.last_lives = 3
            self.last_shots = 0
            
            while not done and steps < MAX_STEPS:
                # Get action from neural network
                action_probs = neural_network.predict(state.reshape(1, -1))[0]
                action = np.argmax(action_probs)
                
                # Take action in environment
                state, _, done, _, info = self.env.step(action)
                state = self.preprocess_state(state)
                
                # Calculate reward
                reward = self.calculate_reward(info)
                total_reward += reward
                steps += 1
        
        return total_reward / num_episodes
    
    def record_episode(self, neural_network, output_path):
        """
        Record a video of the neural network playing Space Invaders.
        Useful for visualizing how well it's doing!
        """
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
        """
        Clean up resources when we're done.
        Always call this when you're finished with the environment!
        """
        self.env.close() 