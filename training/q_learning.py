import numpy as np
import os
from race_track_env.race_track_env import RaceTrack


env = RaceTrack(track_map='b', render_mode=None, size=10)
n_rows, n_cols, y_dim, x_dim = env.nS
n_actions = env.nA

learning_rate = 0.8
discount_factor = 0.95
exploration_prob = 0.2
epochs = 5000
max_steps =500
Q_table = np.zeros((n_rows, n_cols, y_dim, x_dim, n_actions))

for epoch in range(epochs):
    state, _ = env.reset()
    row, col, y_speed, x_speed = state
    y_speed += 4
    x_speed += 4
    for step in range(max_steps):
        if np.random.rand() < exploration_prob:
            action = np.random.randint(n_actions)
        else:
            action = np.argmax(Q_table[row, col, y_speed, x_speed])
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        n_row, n_col, ny_speed, nx_speed = next_state
        nx_speed += 4
        ny_speed += 4
        if 0 <= n_row < n_rows and 0 <= n_col < n_cols and 0 <= ny_speed < 5 and 0 <= nx_speed < 9:
            Q_table[row, col, y_speed, x_speed, action]+= learning_rate*(
                reward + discount_factor * np.max(Q_table[n_row, n_col, ny_speed, nx_speed]) -
                Q_table[row, col, y_speed, x_speed, action]
            )
        row, col, y_speed, x_speed = n_row, n_col, ny_speed, nx_speed
        if done:
            break

# Use os.path to ensure the Q-table is saved in the training directory
q_table_path = os.path.join(os.path.dirname(__file__), "q_table.npy")
np.save(q_table_path, Q_table)
