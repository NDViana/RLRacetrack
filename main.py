import numpy as np
import os
from race_track_env.race_track_env import RaceTrack


def main():

    # Load Q-table
    # Use os.path to ensure correct path regardless of working directory
    q_table_path = os.path.join(os.path.dirname(__file__), "training", "q_table.npy")
    Q = np.load(q_table_path)

    # Initialize environment with rendering
    env = RaceTrack(track_map='b', render_mode='human', size=10)

    # Run one full episode
    state, _ = env.reset()
    row, col, y_speed, x_speed = state
    y_speed += 4
    x_speed += 4

    done = False
    steps = 0

    while not done and steps < 500:
        action = np.argmax(Q[row, col, y_speed, x_speed])
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        row, col, y_speed, x_speed = next_state
        y_speed += 4
        x_speed += 4
        steps += 1

    env.close()


if __name__ == "__main__":
    main()
