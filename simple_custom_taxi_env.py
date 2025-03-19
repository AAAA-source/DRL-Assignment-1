from collections import deque
import gym
import numpy as np
import importlib.util
import time
from IPython.display import clear_output
import random

class SimpleTaxiEnv():
    def __init__(self, grid_size=5, fuel_limit=50, obstacle_count=5):
        self.grid_size = grid_size
        self.fuel_limit = fuel_limit
        self.current_fuel = fuel_limit
        self.passenger_picked_up = False
        self.obstacle_count = obstacle_count
        
        self.stations = [(0, 0), (0, self.grid_size - 1),
                         (self.grid_size - 1, 0), (self.grid_size - 1, self.grid_size - 1)]
        self.passenger_loc = None
       
        self.obstacles = set()
        self.destination = None

    def generate_obstacles(self):
        """Generates obstacles and ensures there's a valid path"""
        while True:
            self.obstacles = set()
            available_positions = [
                (x, y) for x in range(self.grid_size)
                for y in range(self.grid_size)
                if (x, y) not in self.stations
            ]
            self.obstacles = set(random.sample(available_positions, self.obstacle_count))
            
            if self.check_valid_path():  # 確保有通行路徑
                break

    def check_valid_path(self):
        """Check if there’s a valid path from taxi to passenger to destination"""
        visited = set()
        queue = deque([self.taxi_pos])
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            if current == self.passenger_loc or current == self.destination:
                return True

            # Possible moves
            x, y = current
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if (nx, ny) not in self.obstacles and (nx, ny) not in visited:
                        queue.append((nx, ny))
        return False

    def reset(self):
        """Reset the environment"""
        self.current_fuel = self.fuel_limit
        self.passenger_picked_up = False
        
        available_positions = [
            (x, y) for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in self.stations
        ]

        self.taxi_pos = random.choice(available_positions)
        self.passenger_loc = random.choice(self.stations)
        
        possible_destinations = [s for s in self.stations if s != self.passenger_loc]
        self.destination = random.choice(possible_destinations)
        
        self.generate_obstacles()  # Add obstacles with a valid path check

        return self.get_state(), {}
        
    def step(self, action):
        """Perform an action and update the environment state."""
        taxi_row, taxi_col = self.taxi_pos
        next_row, next_col = taxi_row, taxi_col
        reward = 0

        if action == 0:  # Move Down
            next_row += 1
        elif action == 1:  # Move Up
            next_row -= 1
        elif action == 2:  # Move Right
            next_col += 1
        elif action == 3:  # Move Left
            next_col -= 1

        if action in [0, 1, 2, 3]:  # 移動行為
            if (next_row, next_col) in self.obstacles or not (0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size):
                reward -= 5
            else:
                self.taxi_pos = (next_row, next_col)
                if self.passenger_picked_up:
                    self.passenger_loc = self.taxi_pos
        else:
            if action == 4:  # PICKUP
                if self.taxi_pos == self.passenger_loc:
                    self.passenger_picked_up = True
                    self.passenger_loc = (-1, -1)
                    reward += 10
                else:
                    reward = -10
            elif action == 5:  # DROPOFF
                if self.passenger_picked_up:
                    if self.taxi_pos == self.destination:
                        reward += 50
                        return self.get_state(), reward, True, {}
                    else:
                        reward -= 10
                    self.passenger_picked_up = False
                    self.passenger_loc = self.taxi_pos
                else:
                    reward -= 10

        reward -= 0.1
        self.current_fuel -= 1
        if self.current_fuel <= 0:
            return self.get_state(), reward - 10, True, {}

        return self.get_state(), reward, False, {}

    def get_state(self):
        """Return the current environment state following the agreed-upon format."""
        taxi_row, taxi_col = self.taxi_pos
        passenger_row, passenger_col = self.passenger_loc if not self.passenger_picked_up else (self.grid_size, self.grid_size)
        destination_row, destination_col = self.destination if self.passenger_picked_up else (self.grid_size, self.grid_size)

        obstacle_north = int(taxi_row == 0 or (taxi_row - 1, taxi_col) in self.obstacles)
        obstacle_south = int(taxi_row == self.grid_size - 1 or (taxi_row + 1, taxi_col) in self.obstacles)
        obstacle_east = int(taxi_col == self.grid_size - 1 or (taxi_row, taxi_col + 1) in self.obstacles)
        obstacle_west = int(taxi_col == 0 or (taxi_row, taxi_col - 1) in self.obstacles)

        state = (
            taxi_row, taxi_col,
            passenger_row, passenger_col,
            destination_row, destination_col,
            obstacle_north, obstacle_south, obstacle_east, obstacle_west
        )

        return state


    def render_env(self, taxi_pos, action=None, step=None, fuel=None):
        clear_output(wait=True)

        grid = [['.'] * self.grid_size for _ in range(self.grid_size)]

        # Place stations
        grid[0][0] = 'R'
        grid[0][4] = 'G'
        grid[4][0] = 'Y'
        grid[4][4] = 'B'

        # Place taxi
        ty, tx = taxi_pos
        grid[ty][tx] = '🚖'

        # Place obstacles
        for ox, oy in self.obstacles:
            grid[ox][oy] = 'X'

        print(f"\nStep: {step}")
        print(f"Taxi Position: ({ty}, {tx})")
        print(f"Fuel Left: {fuel}")
        print(f"Last Action: {self.get_action_name(action)}\n")

        for row in grid:
            print(" ".join(row))
        print("\n")

    def get_action_name(self, action):
        actions = ["Move South", "Move North", "Move East", "Move West", "Pick Up", "Drop Off"]
        return actions[action] if action is not None else "None"

import importlib.util
import time

if __name__ == "__main__":
    env = SimpleTaxiEnv(grid_size=5, fuel_limit=5000, obstacle_count=5)
    
    obs, _ = env.reset()  # 初始狀態 (reset 產生的 obs)
    total_reward = 0
    done = False
    step_count = 0

    # 匯入 student_agent.py 並使用 get_action()
    spec = importlib.util.spec_from_file_location("student_agent", "student_agent.py")
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)

    while not done:
        env.render_env(env.taxi_pos, action=None, step=step_count, fuel=env.current_fuel)
        #time.sleep(0.05)  # 控制顯示速度

        # 使用 student_agent.py 的 get_action() 選擇行動
        action = student_agent.get_action(obs)

        # 執行行動，並取得新狀態
        obs, reward, done, _ = env.step(action)  # 每一步的 obs 都會在 step() 中更新
        total_reward += reward
        step_count += 1

    print(f"Test Run Finished in {step_count} steps, Total Reward: {total_reward}")

