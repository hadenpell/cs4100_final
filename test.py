import mlagents
from mlagents_envs.environment import UnityEnvironment as UE

env = UE(file_name='AIUnityProject', seed=1, side_channels=[])

print("Number of teams: ", len(env.behavior_specs))

team1 = list(env.behavior_specs)[0]
team1_spec = env.behavior_specs[team1]


for episode in range(3):
  env.reset()
  decision_steps, terminal_steps = env.get_steps(team1)
  tracked_agent = -1 # -1 indicates not yet tracking
  done = False # For the tracked_agent
  episode_rewards = 0 # For the tracked_agent
  max_steps = 20
  steps = 0
  while not done and steps < max_steps:
    steps += 1
    # Track the first agent we see if not tracking
    # Note : len(decision_steps) = [number of agents that requested a decision]
    if tracked_agent == -1 and len(decision_steps) >= 1:
      tracked_agent = decision_steps.agent_id[0]
    # Generate an action for all agents
    '''This is where we will implement an algorithm for learning, currently this is just random learning'''
    action = spec.action_spec.random_action(len(decision_steps))
    # Set the actions
    env.set_actions(team1, action)
    # Move the simulation forward
    env.step()
    # Get the new simulation results
    decision_steps, terminal_steps = env.get_steps(team1)
    if tracked_agent in decision_steps: # The agent requested a decision
      episode_rewards += decision_steps[tracked_agent].reward
    if tracked_agent in terminal_steps: # The agent terminated its episode
      episode_rewards += terminal_steps[tracked_agent].reward
      done = True
env.close()
print("Closed environment")