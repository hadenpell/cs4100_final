#!/usr/bin/env python
# coding: utf-8

# In[45]:


import mlagents


# In[46]:


try:
    env.close()
except:
    pass


# In[47]:


from mlagents_envs.registry import default_registry


# In[48]:


#@title Select Environment { display-mode: "form" }
env_id = "SoccerTwos" #@param ['Basic', '3DBall', '3DBallHard', 'GridWorld', 'Hallway', 'VisualHallway', 'CrawlerDynamicTarget', 'CrawlerStaticTarget', 'Bouncer', 'SoccerTwos', 'PushBlock', 'VisualPushBlock', 'WallJump', 'Tennis', 'Reacher', 'Pyramids', 'VisualPyramids', 'Walker', 'FoodCollector', 'VisualFoodCollector', 'StrikersVsGoalie', 'WormStaticTarget', 'WormDynamicTarget']


# In[49]:


env = default_registry[env_id].make()


# In[50]:


env.reset()


# In[51]:


# We will only consider the first Behavior
#team 0 
behavior_0 = list(env.behavior_specs)[1]
#team 1
behavior_1 = list(env.behavior_specs)[0]
print(f"Name of the behavior : {behavior_0}")
print(f"Name of the behavior : {behavior_1}")


spec0 = env.behavior_specs[behavior_0]
spec1 = env.behavior_specs[behavior_1]


# In[52]:


print("Number of observations : ", len(spec0.observation_specs))
print("Number of observations : ", len(spec1.observation_specs))

# # Is there a visual observation ?
# # Visual observation have 3 dimensions: Height, Width and number of channels
# vis_obs = any(len(spec.shape) == 3 for spec in spec.observation_specs)
# print("Is there a visual observation ?", vis_obs)


# In[53]:


# Is the Action continuous or multi-discrete ?
if spec0.action_spec.continuous_size > 0:
  print(f"There are {spec0.action_spec.continuous_size} continuous actions")
if spec0.action_spec.is_discrete():
  print(f"There are {spec0.action_spec.discrete_size} discrete actions")


# How many actions are possible ?
# print(f"There are {spec0.action_size} action(s)")

# For discrete actions only : How many different options does each action has ?
# if spec0.action_spec.discrete_size > 0:
#   for action, branch_size in enumerate(spec0.action_spec.discrete_branches):
#     print(f"Action number {action} has {branch_size} different options")


# In[54]:


decision_steps0, terminal_steps0 = env.get_steps(behavior_0)
decision_steps1, terminal_steps1 = env.get_steps(behavior_1)


# In[55]:


env.set_actions(behavior_0, spec0.action_spec.empty_action(len(decision_steps0)))
env.set_actions(behavior_1, spec1.action_spec.empty_action(len(decision_steps1)))


# In[56]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

for index, obs_spec in enumerate(spec0.observation_specs):
  if len(obs_spec.shape) == 3:
    #print("Here is the first visual observation")
    plt.imshow(decision_steps.obs[index][0,:,:,:])
    plt.show()

# for index, obs_spec in enumerate(spec.observation_specs):
#   if len(obs_spec.shape) == 1:
#     print("First vector observations : ", decision_steps.obs[index][0,:])


# In[57]:


#env.step()


# In[60]:


for episode in range(3):
  print(episode)
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
    #print(action)
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


# In[ ]:




