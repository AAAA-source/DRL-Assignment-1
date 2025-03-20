import numpy as np
import pickle
import matplotlib.pyplot as plt
from simple_custom_taxi_env import SimpleTaxiEnv

if not hasattr(np, 'bool8'):
    np.bool8 = bool

# Hyperparameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.99999
MIN_EPSILON = 0.1
NUM_EPISODES = 200000

# Initialize environment
env_config = {"fuel_limit": 5000, "obstacle_count": 5}
env = SimpleTaxiEnv(**env_config)

# Q-table initialization
q_table = {}

# Epsilon-greedy action selection
def get_action(state):
    state = tuple(state)
    if state not in q_table:
        q_table[state] = np.zeros(6)

    if np.random.random() < EPSILON:
        return np.random.choice(6)
    return np.argmax(q_table[state])

reward_per_episode = []

# Q-learning training loop
for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    state = tuple(state)

    done = False
    total_reward = 0
    prev_position = state[:2]
    same_position_counter = 0

    while not done:
        action = get_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = tuple(next_state)

        # 🎯 Reward Shaping
        # 每步移動 -0.1 懲罰
        reward -= 0.1

        # 🚨 Stagnation Penalty (無效循環)
        if next_state[:2] == prev_position:
            same_position_counter += 1
            if same_position_counter >= 3:
                reward -= 200
        else:
            same_position_counter = 0
        prev_position = next_state[:2]

        # 🧭 Navigation Guidance
        taxi_pos = state[:2]
        passenger_pos = state[2:4]
        destination_pos = state[4:6]

        if not env.passenger_picked_up:
            dist_before = abs(taxi_pos[0] - passenger_pos[0]) + abs(taxi_pos[1] - passenger_pos[1])
            dist_after = abs(next_state[0] - passenger_pos[0]) + abs(next_state[1] - passenger_pos[1])
            if dist_after < dist_before:
                reward += 1
        else:
            dist_before = abs(taxi_pos[0] - destination_pos[0]) + abs(taxi_pos[1] - destination_pos[1])
            dist_after = abs(next_state[0] - destination_pos[0]) + abs(next_state[1] - destination_pos[1])
            if dist_after < dist_before:
                reward += 1

        # ❌ Avoiding Mistakes
        if action == 4 and state[:2] != state[2:4]:  # Incorrect PICKUP
            reward -= 50
        if action == 5 and state[:2] != state[4:6]:  # Incorrect DROPOFF
            reward -= 50

        # 🚧 Obstacle Collision Penalty
        if action in [0, 1, 2, 3] and next_state[-4 + action]:
            reward -= 100

        # 初始化 next_state 在 Q-table 中的值
        if next_state not in q_table:
            q_table[next_state] = np.zeros(6)

        # Q-value update
        best_next_action = np.argmax(q_table[next_state])
        q_table[state][action] = q_table[state][action] + \
            LEARNING_RATE * (reward + DISCOUNT_FACTOR * q_table[next_state][best_next_action] - q_table[state][action])

        state = next_state
        total_reward += reward

    # Decay epsilon
    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
    
    reward_per_episode.append(total_reward)
    
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}/{NUM_EPISODES}, Total Reward: {total_reward:.2f}, Epsilon: {EPSILON:.3f}")

# Save the Q-table
with open('q_table.pkl', 'wb') as f:
    pickle.dump(q_table, f, protocol=pickle.HIGHEST_PROTOCOL)

# Plot reward curve
plt.plot(np.convolve(reward_per_episode, np.ones(100)/100, mode='valid'))
plt.title('Average Reward per 100 Episodes')
plt.xlabel('Episode (x100)')
plt.ylabel('Average Reward')
plt.grid(True)
plt.show()

print("Training complete. Q-table saved as 'q_table.pkl'.")
