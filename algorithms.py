import numpy as np
import random

class semi_gradient_sarsa:
    """Semi-gradient SARSA algorithm."""
    def __init__(self, env, num_episodes, gamma, epsilon, step_size, get_features, num_features, num_branches, num_actions):
        """
        Initialize Semi-gradient SARSA algorithm

        Args:
            env: a Unity environment
            num_episodes: Number of episodes to take
            gamma: Discount factor
            epsilon: epsilon for epsilon greedy
            step_size: step size
            get_features: returns the features based on a state and action
            num_features: The number of features in the feature vector
            num_actions: Number of actions available for the agent to take
        """
        self.env = env
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.epsilon = epsilon
        self.step_size = step_size
        self.get_features = get_features
        self.num_features = num_features
        self.num_branches = num_branches
        self.num_actions = num_actions
        self.reset_weights()

    def reset_weights(self):
        #one vector of weights for each action
        self.weights = []
        for branch in range(self.num_branches)
            branch_weights = []
            for action in range(self.num_actions):
                branch_weights.append(np.zeros(self.num_features))
            self.weights.append(branch_weights)

    def get_action(self, state, num_agents, spec):
        if np.random.random() < self.epsilon:
            action = []
            for i in (self.num_branches):
                action.append(random.randInt(0, (self.num_actions-1)))
            return action
        else:
            ...

    def q_hat(self, state, action):
        ... ## WEIGHT DOT PRODUCT OF FEATURE VECTOR

    def grad_q_hat(self, state, action):
        return self.get_features(state, action)