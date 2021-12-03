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
            for _ in (self.num_branches):
                action.append(random.randInt(0, (self.num_actions-1)))
            return action
        else:
            action = []
            for b in (self.num_branches):
                action_scores = []
                for a in range(self.num_actions)
                    action_scores.append(self.q_hat(state, b, a))
                bestScore = max(action_scores)
                bestIndices = [index for index in range(len(action_scores)) if action_scores[index] == bestScore]
                action.append(random.choice(bestIndices))
            return action

    def q_hat(self, state, branch, action):
        return self.weights[branch][action].dot(self.get_features(state, branch, action))
        # WE WILL NEED TO CHANGE HOW TO INCORPARATE THE ACTION INTO THE GET FEATURES. HERE IT NEEDS TO BE 1D VECTOR

    #WILL FEATURES BE WITH A STATE, BRANCH, ACTION OR A STATE ACTION WHERE ACTION IS A TUPLE, VERY CONFUSING HARD TO TELL
    def grad_q_hat(self, state, action):
        return self.get_features(state, branch, action)