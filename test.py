#Generates random actions for all 4 agents
import mlagents
from mlagents_envs.registry import default_registry

try:
    env.close()
except:
    pass

#@title Select Environment { display-mode: "form" }
env_id = "SoccerTwos"
env = default_registry[env_id].make()
env.reset()

#team 0 
behavior_0 = list(env.behavior_specs)[1]
#team 1
behavior_1 = list(env.behavior_specs)[0]
print(f"Name of the behavior : {behavior_0}")
print(f"Name of the behavior : {behavior_1}")

spec0 = env.behavior_specs[behavior_0]
spec1 = env.behavior_specs[behavior_1]

decision_steps0, terminal_steps0 = env.get_steps(behavior_0)
decision_steps1, terminal_steps1 = env.get_steps(behavior_1)

env.set_actions(behavior_0, spec0.action_spec.empty_action(len(decision_steps0)))
env.set_actions(behavior_1, spec1.action_spec.empty_action(len(decision_steps1)))


for episode in range(3):
  env.reset()
  decision_steps0, terminal_steps0 = env.get_steps(behavior_0)
  tracked_agent = -1 # -1 indicates not yet tracking
  done = False # For the tracked_agent
  episode_rewards = 0 # For the tracked_agent
  while not done:
    # Track the first agent we see if not tracking
    # Note : len(decision_steps) = [number of agents that requested a decision]
    if tracked_agent == -1 and len(decision_steps0) >= 1:
      tracked_agent = decision_steps0.agent_id[0]

    # Generate an action for all agents
    action = spec0.action_spec.random_action(len(decision_steps0))

    # Set the actions
    env.set_actions(behavior_0, action)
    env.set_actions(behavior_1, action)

    # Move the simulation forward
    env.step()

    # Get the new simulation results
    decision_steps0, terminal_steps0 = env.get_steps(behavior_0)
    decision_steps1, terminal_steps1 = env.get_steps(behavior_1)
    
    if tracked_agent in decision_steps0: # The agent requested a decision
      episode_rewards += decision_steps0[tracked_agent].reward
    if tracked_agent in terminal_steps0: # The agent terminated its episode
      episode_rewards += terminal_steps0[tracked_agent].reward
      done = True
  print(f"Total rewards for episode {episode} is {episode_rewards}")