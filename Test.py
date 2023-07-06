# Evaluate the trained agent on unseen data

# Define the evaluation loop
eval_episodes = 100
eval_rewards = []

for _ in range(eval_episodes):
    state = np.random.rand(1, state_size)  # Random initial state
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state)
        next_state = np.random.rand(1, state_size)  # Random next state
        reward = np.random.randn()  # Random reward
        total_reward += reward
        done = np.random.choice([False, True])  # Random termination

        state = next_state

    eval_rewards.append(total_reward)

average_reward = np.mean(eval_rewards)
print(f"Average Reward on Unseen Data: {average_reward}")

import pickle

# Save the model to a file
with open('trained_model_AT_RL.pkl', 'wb') as file:
    pickle.dump(agent.model, file)