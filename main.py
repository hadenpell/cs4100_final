import mlagents
from mlagents_envs.environment import UnityEnvironment as UE

if __name__ == "__main__":
    env = UE(file_name='AIUnityProject', seed=1, side_channels=[])

    state = env.reset()
    print(state)