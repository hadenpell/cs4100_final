import numpy as np

import mlagents

from matplotlib import pyplot as plt

from algorithms import semi_gradient_sarsa
from mlagents_envs.environment import UnityEnvironment as UE

if __name__ == "__main__":
    
    # Learning Parameters
    MAX_EPISODES = 1_000
    MAX_STEPS    = 100
    GAMMA        = .99
    START_EPSILON      = .6
    END_EPSILON = .1
    STEP_SIZE    = .01
    ######GET FEATURES
    NUM_FEATURES = 21
    NUM_BRANCHES = 2
    NUM_ACTIONS = 3

   

    model = semi_gradient_sarsa('NEW_build_originalFeatures_addedVelocity/UnityEnvironment', MAX_EPISODES, MAX_STEPS, GAMMA, START_EPSILON, END_EPSILON, STEP_SIZE, 'placeholder', NUM_FEATURES, NUM_BRANCHES, NUM_ACTIONS)
    model.weights = np.array([np.array([[-2.98819484e+03, -3.44337784e+01, -1.93713004e+02, -8.90088046e+02,
                                -3.47400632e+01, -1.06542620e+02, -1.06866649e+03, -4.05584787e+02,
                                    1.46345777e+03, -1.68461803e+02,  6.08732846e-01,  8.17281412e+01,
                                -5.83585121e+01,  5.66248782e+01, -6.70531609e+01, -2.59054662e+02,
                                -6.64015266e+01,  2.58034425e+02],
                                [-2.98801455e+03, -3.47572765e+01, -1.93914454e+02, -8.90213965e+02,
                                -3.47752182e+01, -1.06279952e+02, -1.06796841e+03, -4.05266436e+02,
                                    1.46347914e+03, -1.68278940e+02,  2.35695603e-01,  8.14567935e+01,
                                -5.79004915e+01,  5.72930949e+01, -6.64273977e+01, -2.59435693e+02,
                                -6.68921696e+01,  2.57814868e+02],
                                [-2.98778050e+03, -3.38540237e+01, -1.94137774e+02, -8.89566548e+02,
                                -3.45283282e+01, -1.06274171e+02, -1.06829640e+03, -4.05130440e+02,
                                    1.46308603e+03, -1.68461174e+02,  8.74369881e-01,  8.14671079e+01,
                                -5.83974602e+01,  5.64144904e+01, -6.67015733e+01, -2.59092213e+02,
                                -6.61066733e+01,  2.58111647e+02]]),
 
                                np.array([[-2.26549523e+03, -2.20017003e+01, -1.94948183e+02, -4.25920102e+02,
                                -2.19308428e+01, -5.36827079e+01, -9.18194176e+02, -3.91222988e+02,
                                    1.22949182e+03, -1.48559260e+02,  8.00976894e-02,  2.72078024e+01,
                                    2.05337115e+01,  5.91429384e+01, -1.23305303e+02, -1.86614411e+02,
                                    2.20164612e+01 , 1.21981371e+02],
                                [-2.26590598e+03, -2.13137318e+01, -1.95318967e+02, -4.26176526e+02,
                                -2.19945628e+01, -5.38290693e+01, -9.18440705e+02, -3.91561324e+02,
                                    1.22904717e+03, -1.48400388e+02,  9.01230398e-01,  2.74102217e+01,
                                    2.10925496e+01,  5.93481890e+01, -1.23799795e+02, -1.86828562e+02,
                                    2.15547715e+01,  1.21632093e+02],
                                [-2.26593115e+03, -2.18850712e+01, -1.95294850e+02, -4.25759016e+02,
                                -2.19505366e+01, -5.42824070e+01, -9.18096282e+02, -3.91070450e+02,
                                    1.22889091e+03, -1.47717063e+02,  7.85783211e-01,  2.72221658e+01,
                                    2.09543845e+01,  5.98326293e+01, -1.23781821e+02, -1.86407451e+02,
                                    2.12794972e+01,  1.22179957e+02]])])

    steps_per_episode, rewards_per_episode, weights = model.perform_learning()

    print("Here are the trained weights")
    print(weights)

    plt.plot(steps_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Steps per episode: epsilon .6 to .1")
    plt.show()

    plt.plot(rewards_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Rewards per episode: epsilon .6 to .1")
    plt.show()


