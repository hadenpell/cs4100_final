import mlagents

from algorithms import semi_gradient_sarsa
from mlagents_envs.environment import UnityEnvironment as UE

if __name__ == "__main__":
    
    # Learning Parameters
    MAX_EPISODES = 2_000
    MAX_STEPS    = 500
    GAMMA        = .99
    EPSILON      = .01
    STEP_SIZE    = .001
    ######GET FEATURES
    NUM_FEATURES = 9
    NUM_BRANCHES = 3
    NUM_ACTIONS = 3

   

    model = semi_gradient_sarsa(MAX_EPISODES, MAX_STEPS, GAMMA, EPSILON, STEP_SIZE, 'placeholder', NUM_FEATURES, NUM_BRANCHES, NUM_ACTIONS)

    model.perform_learning()
