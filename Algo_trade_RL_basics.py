# %% [markdown]
# Algortihmci trading Model via Reinforcemnt learning using DQN algortihm
# 

# %%
import numpy as np
import tensorflow as tf

# %%
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995  # Decay rate for exploration
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = np.random.choice(len(self.memory), batch_size, replace=False)
        for idx in minibatch:
            state, action, reward, next_state, done = self.memory[idx]
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

# %%

# Main program

# Define the environment and state/action space
state_size = 10
action_size = 3

# Create an instance of the DQNAgent
agent = DQNAgent(state_size, action_size)

# Define the training loop
episodes = 1000
batch_size = 32

for episode in range(episodes):
    state = np.random.rand(1, state_size)  # Random initial state
    total_reward = 0
    done = False
    
    while not done:
        action = agent.act(state)
        next_state = np.random.rand(1, state_size)  # Random next state
        reward = np.random.randn()  # Random reward
        total_reward += reward
        done = np.random.choice([False, True])  # Random termination

        agent.remember(state, action, reward, next_state, done)
        state = next_state

    # Perform experience replay
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

    print(f"Episode: {episode+1}, Total Reward: {total_reward}")

# %%
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

# %%
import pickle

# Save the model to a file
with open('trained_model_AT_RL.pkl', 'wb') as file:
    pickle.dump(agent.model, file)


# %%
import pickle

# Load the saved model from file
with open('trained_model_AT_RL.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Now you can use the loaded model for predictions or further training


# %%
state1=np.random.rand(1,state_size)
action1 = loaded_model.predict(state1)

print(f'state_random={state1}')
print(f'action_probs={action1}')

print(f'action_index_predicted_for_max_rewards={np.argmax(action1)}')


