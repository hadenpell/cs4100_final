import numpy as np
import random

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

    def get_action(self, state, num_agents, spec):
        if np.random.random() < self.epsilon:
            action = []
            for _ in range(self.num_branches):
                action.append(random.randint(0, (self.num_actions-1)))
            return tuple(action)
        else:
            action = []
            for b in range(self.num_branches):
                action_scores = []
                for a in range(self.num_actions):
                    action_scores.append(self.q_hat(state, b, a))
                bestScore = max(action_scores)
                bestIndices = [index for index in range(len(action_scores)) if action_scores[index] == bestScore]
                action.append(random.choice(bestIndices))
            return tuple(action)

    def q_hat(self, state, branch, action):
        return self.weights[branch][action].dot(self.get_features(state, branch, action))
        # WE WILL NEED TO CHANGE HOW TO INCORPARATE THE ACTION INTO THE GET FEATURES. HERE IT NEEDS TO BE 1D VECTOR

    #WILL FEATURES BE WITH A STATE, BRANCH, ACTION OR A STATE ACTION WHERE ACTION IS A TUPLE, VERY CONFUSING HARD TO TELL
    def grad_q_hat(self, state, action):
        return self.get_features(state, branch, action)

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
                """
                #######################
                """

                #determine actions for each team
                team1_action = self.get_action([], 1, TEAM1_SPEC)
                #Set actions
                self.env.set_actions(TEAM1, team1_action)
                #Move simulation forward
                self.env.step()
                # Check new simulation values
                decision_steps, terminal_steps = self.env.get_steps(TEAM1)

                if tracked_agent in terminal_steps:
                    episode_rewards += terminal_steps[tracked_agent].reward
                    print("Our agent received a total reward of ", episode_rewards, " for episode: ", episode)
                    #####UPDATE WEIGHT VALUES
                    done = True
                if tracked_agent in decision_steps: 
                    episode_rewards += decision_steps[tracked_agent].reward
                    #####UPDATE WEIGHT VALUES
                step += 1
        self.env.close()
        