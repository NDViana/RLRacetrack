import os

import numpy as np
from gymnasium import Env
from gymnasium import spaces

import pygame  # pygame is used for rendering

STARTING = 0.8
FINISHING = 0.4


# Race track environment
class RaceTrack(Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 20}

    def __init__(self, track_map: str, render_mode: str = None, size: int = 2):
        self.size = size  # the size of cells

        assert track_map in ['a', 'b']
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        # reading a track map
        filename = 'track_a.npy' if track_map == 'a' else 'track_b.npy'
        base_path = os.path.dirname(__file__)
        map_path = os.path.join(base_path, 'maps', filename)

        with open(map_path, 'rb') as f:
            self.track_map = np.load(f)

        # Initialize parameters for pygame
        self.window_size = self.track_map.shape
        # Pygame's coordinate if the transpose of that of numpy
        self.window_size = (self.window_size[1] * self.size, self.window_size[0] * self.size)
        self.window = None  # window for pygame rendering
        self.clock = None  # clock for pygame ticks
        self.truncated = False

        # Get start states
        self.start_states = np.dstack(np.where(self.track_map == STARTING))[0]

        # Define proper Gymnasium spaces
        # Observation space: (row, col, y_speed, x_speed)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.track_map.shape[0]),  # row
            spaces.Discrete(self.track_map.shape[1]),  # col
            spaces.Discrete(5),  # y_speed (-4 to 0)
            spaces.Discrete(9)   # x_speed (-4 to 4)
        ))
        # Action space: 9 possible acceleration combinations
        self.action_space = spaces.Discrete(9)

        # Keep these for backward compatibility
        self.nS = (*self.track_map.shape, 5, 9)  # observation space
        self.nA = 9  # action space

        self.state = None  # Initialize state
        self.speed = None  # Initialize speed

        # Mapping the integer action to acceleration tuple
        self._action_to_acceleration = {
            0: (-1, -1),
            1: (-1, 0),
            2: (-1, 1),
            3: (0, -1),
            4: (0, 0),
            5: (0, 1),
            6: (1, -1),
            7: (1, 0),
            8: (1, 1)
        }

    # Get observation
    def _get_obs(self):
        return (*self.state, *self.speed)

    # Get info, always return None in our case
    def _get_info(self):
        return None

    # Check if the race car go accross the finishing line
    def _check_finish(self, state=None):
        if state is None:
            state = self.state
        finish_states = np.where(self.track_map == FINISHING)
        rows = finish_states[0]
        col = finish_states[1][0]
        if state[0] in rows and state[1] >= col:
            return True
        return False

    # Check if the track run out of the track
    def _check_out_track(self, next_state):
        row, col = next_state
        H, W = self.track_map.shape
        # If the car go out of the boundary
        if row < 0 or row >= H or col < 0 or col >= W:
            return True
        # Check if the car run into the gravels
        if self.track_map[next_state[0], next_state[1]] == 0:
            return True

        # Check if part of the path run into gravels
        # Check row path (vertical movement)
        row_step = 1 if row > self.state[0] else -1
        if row_step != 0:  # If there is vertical movement
            for r in range(self.state[0] + row_step, row + row_step, row_step):
                if self.track_map[r, self.state[1]] == 0: return True

        # Check column path (horizontal movement)
        col_step = 1 if col > self.state[1] else -1
        if col_step != 0:  # If there is horizontal movement
            for c in range(self.state[1] + col_step, col + col_step, col_step):
                if self.track_map[row, c] == 0: return True

        return False

    # reset the car to one of the starting positions
    def reset(self, *, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Select start position randomly from the starting line
        start_idx = np.random.choice(self.start_states.shape[0])
        self.state = self.start_states[start_idx]
        self.speed = (0, 0)

        if self.render_mode == 'human':
            self.render()

        return self._get_obs(), self._get_info()

    # take actions
    def step(self, action):
        # Get new acceleration and updated position
        new_state = np.copy(self.state)
        y_act, x_act = self._action_to_acceleration[action]

        temp_y_acc = self.speed[0] + y_act
        temp_x_acc = self.speed[1] + x_act

        if temp_y_acc < -4: temp_y_acc = -4
        if temp_y_acc > 0: temp_y_acc = 0  # Avoid the car from going backward
        if temp_x_acc < -4: temp_x_acc = -4
        if temp_x_acc > 4: temp_x_acc = 4

        new_state[0] += temp_y_acc
        new_state[1] += temp_x_acc

        terminated = False
        reward = -1  # Default reward is -1 for each step
        info = {}

        # check if next position crosses the finishing line
        if self._check_finish(new_state):
            terminated = True
            self.state = new_state
            self.speed = (temp_y_acc, temp_x_acc)
            reward = 0  # No penalty for reaching the finish line
        # check if next position locates in invalid places
        elif self._check_out_track(new_state):
            # Instead of calling reset() which would return values and break the step flow,
            # we'll just reset the state and speed, and add a penalty
            start_idx = np.random.choice(self.start_states.shape[0])
            self.state = self.start_states[start_idx]
            self.speed = (0, 0)
            reward = -10  # Penalty for going out of track
            info['out_of_track'] = True
        else:
            self.state = new_state
            self.speed = (temp_y_acc, temp_x_acc)

        if self.render_mode == 'human':
            self.render()

        return self._get_obs(), reward, terminated, self.truncated, info

    # visualize race map
    def render(self, mode=None):
        if mode is None:
            mode = self.render_mode

        if mode is None:
            return

        if self.window is None:
            print(f"Initializing Pygame window with size: {self.window_size}")
            pygame.init()
            pygame.display.set_caption('Race Track')
            if mode == 'human':
                self.window = pygame.display.set_mode(self.window_size)
                print(f"Pygame window created: {self.window is not None}")
                # Force the window to update and be visible
                pygame.display.flip()
                # Try to bring the window to the foreground
                try:
                    import ctypes
                    # Get the window handle
                    hwnd = pygame.display.get_wm_info()['window']
                    # Bring the window to the foreground
                    ctypes.windll.user32.SetForegroundWindow(hwnd)
                    print("Attempted to bring window to foreground")
                except Exception as e:
                    print(f"Could not bring window to foreground: {e}")

        if self.clock is None:
            self.clock = pygame.time.Clock()

        rows, cols = self.track_map.shape
        # Use a white background as requested
        self.window.fill((255, 255, 255))

        # Draw the map
        for row in range(rows):
            for col in range(cols):
                cell_val = self.track_map[row, col]
                # Draw finishing cells
                if cell_val == FINISHING:
                    fill = (235, 52, 52)
                    pygame.draw.rect(self.window, fill, (col * self.size, row * self.size, self.size, self.size), 0)
                # Draw starting cells
                elif cell_val == STARTING:
                    fill = (61, 227, 144)
                    pygame.draw.rect(self.window, fill, (col * self.size, row * self.size, self.size, self.size), 0)

                color = (0, 0, 0)  # Black outline for all cells
                # Draw gravels
                if cell_val == 0:
                    color = (255, 255, 255)  # White for gravels
                # Draw race track
                elif cell_val == 1:
                    color = (128, 128, 128)  # Gray for race track

                pygame.draw.rect(self.window, color, (col * self.size, row * self.size, self.size, self.size), 1)

        # Draw the car with a blue color as requested
        pygame.draw.rect(self.window, (0, 0, 255),
                         (self.state[1] * self.size, self.state[0] * self.size, self.size, self.size), 0)

        if mode == "human":
            pygame.display.update()
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    print("Received QUIT event, closing window")
                    self.window = None
                    pygame.quit()
                    self.truncated = True
            self.clock.tick(self.metadata['render_fps'])

    def close(self):
        """Clean up resources, particularly the Pygame window."""
        print("Closing environment and cleaning up resources")
        if self.window is not None:
            print("Closing Pygame window")
            pygame.quit()
            self.window = None
            self.clock = None
