import numpy as np
import random

import mlagents
from mlagents_envs.base_env import ActionTuple

class semi_gradient_sarsa:
    """Semi-gradient SARSA algorithm."""
    def __init__(self, env, num_episodes, num_steps, gamma, epsilon, step_size, get_features, num_features, num_branches, num_actions):
        """
        Initialize Semi-gradient SARSA algorithm

        Args:
            env: a Unity environment
            num_episodes: Number of episodes to take
            num_steps: Number of steps maximum for each episode
            gamma: Discount factor
            epsilon: epsilon for epsilon greedy
            step_size: step size
            get_features: returns the features based on a state and action
            num_features: The number of features in the feature vector
            num_actions: Number of actions available for the agent to take
        """
        self.env = env
        self.num_episodes = num_episodes
        self.num_steps = num_steps
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
        for branch in range(self.num_branches):
            branch_weights = []
            for action in range(self.num_actions):
                branch_weights.append(np.zeros(self.num_features))
            self.weights.append(branch_weights)

    def get_action(self, features, num_agents, spec):
        if np.random.random() < self.epsilon:
            action = []
            for b in range(self.num_branches):
                action.append(random.randint(0, (self.num_actions-1)))
            action = np.array([action])
            return ActionTuple(discrete=action)
        else:
            action = []
            for b in range(self.num_branches):
                action_scores = []
                for a in range(self.num_actions):
                    action_scores.append(self.q_hat(features, b, a))
                bestScore = max(action_scores)
                bestIndices = [index for index in range(len(action_scores)) if action_scores[index] == bestScore]
                action.append(random.choice(bestIndices))
            action = np.array([action])
            return ActionTuple(discrete=action)

    def q_hat(self, features, branch, action):
        return self.weights[branch][action].dot(features)
        # WE WILL NEED TO CHANGE HOW TO INCORPARATE THE ACTION INTO THE GET FEATURES. HERE IT NEEDS TO BE 1D VECTOR

    def perform_learning(self):
        self.env.reset()
        TEAM1 = list(self.env.behavior_specs)[0]
        TEAM1_SPEC = self.env.behavior_specs[TEAM1]
        for episode in range(self.num_episodes):
            self.env.reset()
            decision_steps, terminal_steps = self.env.get_steps(TEAM1)
            done = False
            episode_rewards = 0 #Rewards of the tracked_agent (US)
            step = 0
            while not done and step < self.num_steps:
                #grab our agent
                tracked_agent = decision_steps.agent_id[0]

                #CREATE FEATURE VECTOR
                features = decision_steps.obs[0][0]

                #determine actions for each team
                team1_action = self.get_action(features, 1, TEAM1_SPEC)
                #Set actions
                self.env.set_actions(TEAM1, team1_action)
                #Move simulation forward
                self.env.step()
                # Check new simulation values
                decision_steps, terminal_steps = self.env.get_steps(TEAM1)

                if tracked_agent in terminal_steps:
                    reward = terminal_steps[tracked_agent].reward
                    episode_rewards += reward
                    print("Our agent received a total reward of ", episode_rewards, " for episode: ", episode)
                    for b in range(self.num_branches):
                        self.weights[b] = self.weights[b] + self.step_size * (reward - self.q_hat(features, b, team1_action.discrete[0][b]))*features
                    done = True
                if tracked_agent in decision_steps:
                    reward = decision_steps[tracked_agent].reward
                    episode_rewards += reward
                    next_features = decision_steps.obs[0][0]
                    next_team1_action = self.get_action(next_features, 1, TEAM1_SPEC)
                    for b in range(self.num_branches):
                        self.weights[b] = self.weights[b] + self.step_size * (reward + self.gamma * self.q_hat(next_features, b, next_team1_action.discrete[0][b])  - self.q_hat(features, b, team1_action.discrete[0][b]))*features
                step += 1
        self.env.close()
        