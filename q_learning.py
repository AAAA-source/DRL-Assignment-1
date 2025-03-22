import numpy as np
import pickle
import matplotlib.pyplot as plt
from simple_custom_taxi_env import SimpleTaxiEnv

# Compatibility fix for NumPy version compatibility
if not hasattr(np, 'bool8'):
    np.bool8 = bool

# Hyperparameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.99998
MIN_EPSILON = 0.1
NUM_EPISODES = 100000

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
oscillation_penalty_per_episode = []
success_count = 0
state_count = 0

# Q-learning training loop
for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    state = tuple(state)

    done = False
    total_reward = 0
    total_step = 0
    total_oscillation_penalty = 0
    success = False  # 新增追蹤成功標誌

    # Tracking conditions
    previously_picked_up = False
    previously_dropped_off = False
    prev_position = state[:2]
    previous_position = None

    while not done:

        action = get_action(state)
        next_state, reward, done = env.step(action)
        next_state = tuple(next_state)

        # 🎯 Reward Shaping
        reward -= 1  # 每步固定懲罰
        total_step += 1

        # 🚨 Stagnation Penalty (原地不動)
        if next_state[:2] == prev_position:
            reward -= 10
            total_oscillation_penalty += 100

        # 🚨 強化來回震盪懲罰 (立即觸發)
        if previous_position == next_state[:2] and prev_position == state[:2]:
            reward -= 2000000000000
            total_oscillation_penalty += 20000

        # 更新位置追蹤
        previous_position = prev_position
        prev_position = next_state[:2]

        # 🧭 Navigation Guidance
        taxi_pos = state[:2]
        passenger_pos = state[2:4]
        destination_pos = state[4:6]

        if not env.passenger_picked_up:
            dist_before = abs(taxi_pos[0] - passenger_pos[0]) + abs(taxi_pos[1] - passenger_pos[1])
            dist_after = abs(next_state[0] - passenger_pos[0]) + abs(next_state[1] - passenger_pos[1])
            if dist_after < dist_before:
                reward += 5000000
            elif dist_after > dist_before:
                reward -= 5000000
        else:
            dist_before = abs(taxi_pos[0] - destination_pos[0]) + abs(taxi_pos[1] - destination_pos[1])
            dist_after = abs(next_state[0] - destination_pos[0]) + abs(next_state[1] - destination_pos[1])
            if dist_after < dist_before:
                reward += 10000000
            elif dist_after > dist_before:
                reward -= 10000000

        # 🚖 正確執行 PICKUP
        if action == 4:
            if state[:2] == passenger_pos and not previously_picked_up and env.passenger_picked_up:
                reward += 100000
                previously_picked_up = True
            elif previously_picked_up:
                reward -= 500000000
            elif state[:2] != passenger_pos:
                reward -= 500000000

        # 🚖 正確執行 DROPOFF
        if action == 5:
            if state[:2] == destination_pos and previously_picked_up and not env.passenger_picked_up:
                reward += 200000
                previously_dropped_off = True
            elif previously_dropped_off:
                reward -= 500000000
            elif state[:2] != destination_pos:
                reward -= 500000000

        # 🚧 Obstacle Collision Penalty
        if action in [0, 1, 2, 3] and next_state == state:
            reward -= 100000
            total_oscillation_penalty += 1000

        if done:
            if success:
                reward += 50000000000000000 # ⭐ 成功完成任務的額外獎勵

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

    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
    #print(f"Episode {episode + 1} ended at step {total_step}, Fuel left: {env.current_fuel}")


    reward_per_episode.append(total_reward)
    step_per_episode.append(total_step)
    oscillation_penalty_per_episode.append(total_oscillation_penalty)
    if env.current_fuel > 0 :
        #print("success ! count + 1")
        success_count += 1
    

    if (episode + 1) % 100 == 0:
        average_step = np.mean(step_per_episode[-100:])
        average_reward = np.mean(reward_per_episode[-100:])
        average_oscillation_penalty = np.mean(oscillation_penalty_per_episode[-100:])
        print(f"Episode {episode + 1}/{NUM_EPISODES}, "
              f"Total Reward: {average_reward:.2f}, "
              f"Epsilon: {EPSILON:.3f}, "
              f"Average Step: {average_step}, "
              f"Avg Oscillation Penalty: {average_oscillation_penalty}, "
              f"Success Rate: {success_count / episode + 1}%")

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

# 最終結果
print("Training complete. Q-table saved as 'q_table.pkl'.")
print(f"Total states explored: {state_count}")
print(f"Average steps per episode: {np.mean(step_per_episode):.2f}")
print(f"Average oscillation penalty per episode: {np.mean(oscillation_penalty_per_episode):.2f}")
