import mlagents

from mlagents_envs.environment import UnityEnvironment as UE

if __name__ == "__main__":
    env = UE(file_name='AIUnityProject', seed=1, side_channels=[])
    
    # Learning Parameters
    MAX_EPISODES = 10
    MAX_STEPS    = 2000

    # Environment Constants
    env.reset()
    TEAM1 = list(env.behavior_specs)[0]
    TEAM1_SPEC = env.behavior_specs[TEAM1]

    for episode in range(MAX_EPISODES):
        env.reset()
        decision_steps, terminal_steps = env.get_steps(TEAM1)
        done = False
        episode_rewards = 0 # Rewards of the tracked_agent (US)
        step = 0
        while not done and step < MAX_STEPS:
            #print("Current Episode: ", episode, " Current Step: ", step)

            # Grab our agent
            tracked_agent = decision_steps.agent_id[0]
            # Determine actions for each team
            team1_action = TEAM1_SPEC.action_spec.random_action(len(decision_steps))
            # Set actions
            env.set_actions(TEAM1, team1_action)
            # Move simulation forward
            env.step()
            # Check new simulation
            decision_steps, terminal_steps = env.get_steps(TEAM1)
            if tracked_agent in decision_steps: # The agent request a decision
                episode_rewards += decision_steps[tracked_agent].reward
            if tracked_agent in terminal_steps: # The game has reached a terminal state
                episode_rewards += terminal_steps[tracked_agent].reward
                done = True
                print("Our agent received a total reward of ", episode_rewards, " for episode: ", episode)
            step += 1 
    env.close()
    print("Closed Environment")
