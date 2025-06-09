import numpy as np
import matplotlib.pyplot as plt

# This module contains functions to build race track layouts for reinforcement learning environments.
# The tracks are represented as 2D numpy arrays with the following values:
# - 0: Off-track area (agent cannot move here)
# - 1: Valid track area (agent can move here)
# - STARTING (0.8): Starting line position
# - FINISHING (0.4): Finish line position

STARTING = 0.8 # the value of the starting line
FINISHING = 0.4 # the value of the finishing line

def build_track_a(save_map=False):
    """
    Creates track layout A - a simple oval-like track with dimensions 32x17.

    Args:
        save_map (bool): If True, saves the track to a .npy file

    Returns:
        numpy.ndarray: 2D array representing the track layout
    """
    # Initialize track with all valid areas (1s)
    track = np.ones(shape=(32,17))

    # Define off-track areas (0s) to create the track shape
    # These lines carve out the inner and outer boundaries of the track
    track[14:, 0] = 0
    track[22:, 1] = 0
    track[-3:, 2] = 0
    track[:4, 0] = 0
    track[:3, 1] = 0
    track[0, 2] = 0
    track[6:, -8:] = 0

    # Ensure specific track point is valid (for track connectivity)
    track[6, 9] = 1

    # Set finishing line on the right edge of the top section
    track[:6, -1] = FINISHING

    # Set starting line on the bottom edge
    track[-1, 3:9] = STARTING

    # Save the track to a file if requested
    if save_map:
        with open('track_a.npy', 'wb') as f:
            np.save(f, track)
    return track

def build_track_b(save_map=False):
    """
    Creates track layout B - a more complex track with dimensions 30x32.
    This track has a diagonal section and more intricate boundaries.

    Args:
        save_map (bool): If True, saves the track to a .npy file

    Returns:
        numpy.ndarray: 2D array representing the track layout
    """

    # Initialize track with all valid areas (1s)
    track = np.ones(shape=(30,32))

    # Create a diagonal boundary on the left side of the track
    # This loop creates a progressively larger off-track area moving from right to left
    for i in range(14):
        track[:(-3 - i), i] = 0  # Top-left diagonal boundary

    # Create a corridor/passage in the middle section of the track
    track[3:7, 11] = 1   # Ensure these specific areas are valid track
    track[2:8, 12] = 1   # Widening the corridor
    track[1:9, 13] = 1   # Further widening the corridor

    # Define additional off-track areas to shape the track
    track[0, 14:16] = 0
    track[-17:, -9:] = 0
    track[12, -8:] = 0
    track[11, -6:] = 0
    track[10, -5:] = 0
    track[9, -2:] = 0

    # Set starting line on the bottom edge (only where track is valid)
    track[-1] = np.where(track[-1] == 0, 0, STARTING)

    # Set finishing line on the right edge (only where track is valid)
    track[:, -1] = np.where(track[:, -1] == 0, 0, FINISHING)

    # Save the track to a file if requested
    if save_map:
        with open('track_b.npy', 'wb') as f:
            np.save(f, track)
    return track


if __name__ == "__main__":
    # create tracks and save them as .npy files
    track_a = build_track_a(save_map=True)
    track_b = build_track_b(save_map=True)

    # check if the map properly built
    plt.figure(figsize=(10, 5), dpi=150)
    for i, map_type in enumerate(['a', 'b']):
        with open(f'track_{map_type}.npy', 'rb') as f:
            track = np.load(f)
        ax = plt.subplot(1, 2, i + 1)
        ax.imshow(track, cmap='GnBu')
        ax.set_title(f'track {map_type}', fontdict={'fontsize': 13, 'fontweight': 'bold'})

    plt.tight_layout()
    plt.savefig('maps.png')
    plt.show()
