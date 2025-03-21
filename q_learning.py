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
step_per_episode = []
state_count = 0

# Q-learning training loop
for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    state = tuple(state)

    done = False
    total_reward = 0
    prev_position = state[:2]
    same_position_counter = 0
    total_step = 0

    while not done:
        action = get_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = tuple(next_state)

        # 🎯 Reward Shaping
        # 每步移動 -0.1 懲罰
        reward -= 0.1
        total_step += 1

        # 🚨 Stagnation Penalty (無效循環) — 強化懲罰
        if next_state[:2] == prev_position:
            same_position_counter += 1
            reward -= 100  # 每次原地不動懲罰
            if same_position_counter >= 2:
                reward -= 500
        else:
            same_position_counter = 0
        prev_position = next_state[:2]

        # 🧭 Navigation Guidance (前進與遠離目標的懲罰/獎勵相等)
        taxi_pos = state[:2]
        passenger_pos = state[2:4]
        destination_pos = state[4:6]

        if not env.passenger_picked_up:
            dist_before = abs(taxi_pos[0] - passenger_pos[0]) + abs(taxi_pos[1] - passenger_pos[1])
            dist_after = abs(next_state[0] - passenger_pos[0]) + abs(next_state[1] - passenger_pos[1])
            if dist_after < dist_before:
                reward += 10
            elif dist_after > dist_before:
                reward -= 15  # ⚠️ 與「接近」的獎勵一致，避免原地刷分
        else:
            dist_before = abs(taxi_pos[0] - destination_pos[0]) + abs(taxi_pos[1] - destination_pos[1])
            dist_after = abs(next_state[0] - destination_pos[0]) + abs(next_state[1] - destination_pos[1])
            if dist_after < dist_before:
                reward += 10
            elif dist_after > dist_before:
                reward -= 15  # ⚠️ 與「接近」的獎勵一致，避免原地刷分

        # 🚖 正確執行 PICKUP
        if action == 4:  # PICKUP
            if state[:2] == state[2:4] and not env.passenger_picked_up:
                reward += 50  # ✅ 正確執行 PICKUP
            elif state[:2] != state[2:4]:
                reward -= 100  # ❌ 錯誤執行 PICKUP

        # 🚨 應該 PICKUP 卻未執行 (走過頭)
        if not env.passenger_picked_up and state[:2] == state[2:4] and action != 4:
            reward -= 100  # ❗ 應該 PICKUP 卻沒做

        # 🚖 正確執行 DROPOFF
        if action == 5:  # DROPOFF
            if state[:2] == state[4:6] and env.passenger_picked_up:
                reward += 100  # ✅ 正確執行 DROPOFF
            elif state[:2] != state[4:6]:
                reward -= 100  # ❌ 錯誤執行 DROPOFF

        # 🚨 應該 DROPOFF 卻未執行 (走過頭)
        if env.passenger_picked_up and state[:2] == state[4:6] and action != 5:
            reward -= 100  # ❗ 應該 DROPOFF 卻沒做

        # 🚧 Obstacle Collision Penalty
        if action in [0, 1, 2, 3] and next_state[-4 + action]:
            reward -= 100
            
        if done :
            reward += 1000
            

        # 初始化 next_state 在 Q-table 中的值
        if next_state not in q_table:
            q_table[next_state] = np.zeros(6)
            state_count += 1

        # Q-value update
        best_next_action = np.argmax(q_table[next_state])
        q_table[state][action] = q_table[state][action] + \
            LEARNING_RATE * (reward + DISCOUNT_FACTOR * q_table[next_state][best_next_action] - q_table[state][action])

        state = next_state
        total_reward += reward


    # Decay epsilon
    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
    
    reward_per_episode.append(total_reward)
    step_per_episode.append(total_step)
    
    if (episode + 1) % 100 == 0:
        average_step = np.mean( step_per_episode[-100 : ] )
        average_reward = np.mean( reward_per_episode[-100 : ] )
        print(f"Episode {episode + 1}/{NUM_EPISODES}, Total Reward: {average_reward:.2f}, Epsilon: {EPSILON:.3f} , Average Step : {average_step}")

# Save the Q-table
with open('q_table.pkl', 'wb') as f:
    pickle.dump(q_table, f, protocol=pickle.HIGHEST_PROTOCOL)

# Plot reward curve
plt.plot(np.convolve(reward_per_episode, np.ones(100)/100, mode='valid'))
plt.title('Reward per Episodes')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.grid(True)
plt.show()

print("Training complete. Q-table saved as 'q_table.pkl'.")
print(f"total state : {state_count}") ;
