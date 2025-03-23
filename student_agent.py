import pickle
import numpy as np
import random

# 嘗試多種方法加載 Q-table
try:
    with open('q_table.pkl', 'rb') as f:
        q_table = pickle.load(f, encoding='latin1')
        if not isinstance(q_table, dict):
            raise ValueError("Invalid Q-table format")
except (FileNotFoundError, EOFError, pickle.UnpicklingError, ValueError):
    print("Q-table not found or corrupted. Using random actions only.")
    q_table = {}


def get_action(obs , epsilon = 0.0):
    """根據已學到的 Q-table 做行為選擇，並考慮未知狀態的備案策略。"""
    state = tuple(obs)

    if state not in q_table:
        return random.choice([0, 1, 2, 3, 4, 5])
    
    #if np.random.random() < 0.2:
    #    action = np.random.choice(6)  # Explore

    action = np.argmax(q_table[state])
    return int(action) if action in range(6) else random.choice([0, 1, 2, 3])
