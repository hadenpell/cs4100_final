import numpy as np

import mlagents

from features import create_get_features
from matplotlib import pyplot as plt

from algorithms import semi_gradient_sarsa
from mlagents_envs.environment import UnityEnvironment as UE

if __name__ == "__main__":
    
    # Learning Parameters
    MAX_EPISODES = 1_000
    MAX_STEPS    = 100
    GAMMA        = .99
    START_EPSILON      = .8
    END_EPSILON = .01
    EPSILON_END_STEP = 500
    STEP_SIZE    = .01
    ######GET FEATURES
    NUM_FEATURES = 360012
    NUM_BRANCHES = 2
    NUM_ACTIONS = 3
    FEATURE_VECTOR = "best"

   

    model = semi_gradient_sarsa('NEW_build_originalFeatures_addedVelocity/UnityEnvironment', MAX_EPISODES, MAX_STEPS, GAMMA, START_EPSILON, END_EPSILON, EPSILON_END_STEP, STEP_SIZE, create_get_features(FEATURE_VECTOR), NUM_FEATURES, NUM_BRANCHES, NUM_ACTIONS)


    steps_per_episode, rewards_per_episode, weights = model.perform_learning()

    print("Here are the trained weights")
    print(weights)

    plt.plot(steps_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("onebyone_onehot: Steps per episode")
    plt.show()

    plt.plot(rewards_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("onebyone_onehot: Rewards per episode")
    plt.show()


