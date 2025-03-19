import numpy as np
import pickle
import random

# 載入 Q-table
try:
    with open('q_table.pkl', 'rb') as f:
        q_table = pickle.load(f, encoding='latin1')  # 加入 encoding 以解決 NumPy 相容問題
except (FileNotFoundError, EOFError, pickle.UnpicklingError):
    print("Q-table not found or corrupted. Using random actions only.")
    q_table = {}  # 無 Q-table 時預設為空字典

def get_action(obs):
    """根據已學到的 Q-table 做行為選擇，並考慮未知狀態的備案策略。"""
    state = tuple(obs)  # obs 本身已是 `get_state()` 結果，因此轉為 tuple 即可

    # 若 state 不在 Q-table，執行隨機行為 (避免無限循環)
    if state not in q_table:
        return random.choice([0, 1, 2, 3])  # 嘗試移動行為，避免直接選擇 Pick-up/Drop-off
    
    # 選擇最佳行為
    action = np.argmax(q_table[state])

    return action
