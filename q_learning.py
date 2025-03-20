import numpy as np
import pickle
import matplotlib.pyplot as plt
from simple_custom_taxi_env import SimpleTaxiEnv

if not hasattr(np, 'bool8'):
    np.bool8 = bool  # 將 `np.bool8` 替換為 `bool`

# Hyperparameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON = 1.0         # Initial exploration rate
EPSILON_DECAY = 0.99995 # Decay rate for epsilon
MIN_EPSILON = 0.1     # Minimum epsilon
NUM_EPISODES = 100000  # Number of episodes for training

# Initialize environment
env_config = {
    "fuel_limit": 5000,
    "obstacle_count": 5
}
env = SimpleTaxiEnv(**env_config)

# Q-table initialization using dictionary
q_table = {}

# Epsilon-greedy action selection
def get_action(state):
    state = tuple(state)  # 將 state 轉為 tuple，確保可作為 dict 的 Key
    if state not in q_table:
        q_table[state] = np.zeros(6)  # 初始化 Q-values 為 0

    if np.random.random() < EPSILON:
        return np.random.choice(6)  # Random action (exploration)
    return np.argmax(q_table[state])  # Best action (exploitation)

reward_per_episode = []
# Q-learning training loop
for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    state = tuple(state)

    done = False
    total_reward = 0
    prev_position = state[:2]  # 儲存上一步的位置
    same_position_counter = 0

    while not done:
        action = get_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = tuple(next_state)

        # 🎯 Reward Shaping
        # 每步移動 -0.1 懲罰
        reward -= 0.1

        # 避免陷入無效循環 (原地不動) 懲罰
        if next_state[:2] == prev_position:
            same_position_counter += 1
            if same_position_counter >= 5:
                reward -= 200
        else:
            same_position_counter = 0
        prev_position = next_state[:2]

        # 碰撞障礙物 (X) 懲罰
        if action in [0, 1, 2, 3] and next_state[-4 + action]:  # 上下左右的障礙物對應
            reward -= 50

        # 初始化 next_state 在 Q-table 中的值
        if next_state not in q_table:
            q_table[next_state] = np.zeros(6)

        # Q-value update (Q-learning update rule)
        best_next_action = np.argmax(q_table[next_state])
        q_table[state][action] = q_table[state][action] + \
            LEARNING_RATE * (reward + DISCOUNT_FACTOR * q_table[next_state][best_next_action] - q_table[state][action])

        state = next_state
        total_reward += reward

    # Decay epsilon to reduce exploration over time
    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
    
    reward_per_episode.append(total_reward)
    
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}/{NUM_EPISODES}, Total Reward: {total_reward:.2f}, Epsilon: {EPSILON:.3f}")

# Save the Q-table for inference
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
