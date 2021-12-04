import mlagents

from algorithms import semi_gradient_sarsa
from mlagents_envs.environment import UnityEnvironment as UE

if __name__ == "__main__":
    env = UE(file_name='AIUnityProject', seed=1, side_channels=[])
    
    # Learning Parameters
    MAX_EPISODES = 10
    MAX_STEPS    = 1_500_000
    GAMMA        = .99
    EPSILON      = .1
    STEP_SIZE    = .1
    ######GET FEATURES
    NUM_FEATURES = 9
    NUM_BRANCHES = 3
    NUM_ACTIONS = 3

    model = semi_gradient_sarsa(env, MAX_EPISODES, MAX_STEPS, GAMMA, EPSILON, STEP_SIZE, 'placeholder', NUM_FEATURES, NUM_BRANCHES, NUM_ACTIONS)

    model.perform_learning()
