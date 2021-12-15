import mlagents
from mlagents_envs.environment import UnityEnvironment as UE

import numpy as np

if __name__ == "__main__":
  env = UE("NEW_build_originalFeatures_addedVelocity/UnityEnvironment", seed=1, side_channels=[], worker_id=1, no_graphics=True)
  env.reset()
  env.obse\

