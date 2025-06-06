import numpy as np

# Simple grid environment (4 states, 2 actions)
n_states = 4
n_actions = 2
q_table = np.zeros((n_states, n_actions))

# Hyperparameters
alpha = 0.1      # Learning rate
gamma = 0.9      # Discount factor
epsilon = 0.2    # Exploration rate
episodes = 100

# Simple reward and transition logic
def step(state, action):
    if state == 0:
        return (1, 1) if action == 1 else (0, 0)
    elif state == 1:
        return (2, 1) if action == 1 else (1, 0)
    elif state == 2:
        return (3, 10) if action == 1 else (2, 0)
    else:
        return (3, 0)  # Terminal state

for episode in range(episodes):
    state = 0
    done = False
    while not done:
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = np.random.choice(n_actions)
        else:
            action = np.argmax(q_table[state])

        next_state, reward = step(state, action)

        # Q-learning update
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )

        if next_state == 3:
            done = True
        state = next_state

print("Trained Q-table:")
print(q_table)