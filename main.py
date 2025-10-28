import pygame
import numpy as np
import cv2
from map import Map
from robot import Robot
from visual import *
import json

map = "maps/map.json" # small map, 5 robots
# map = "maps/maze.json" # large maze, 8 robots

# strategy = "closest_frontier"
# strategy = "auction"
strategy = "competitive"

animation = True
record = False

# -------------------- Helper Functions --------------------
def load_config(path):
    """Load environment and robot configuration from JSON file."""
    with open(path, "r") as f:
        cfg = json.load(f)

    GRID_W = cfg["grid_w"]
    GRID_H = cfg["grid_h"]
    CELL_SIZE = cfg["cell_size"]
    SENSE_RADIUS = cfg["sense_radius"]
    COMM_RANGE = cfg["comm_range"]
    OBSTACLES = [tuple(pos) for pos in cfg["obstacles"]]
    ROBOTS_POS = [tuple(pos) for pos in cfg["robots"]]

    set_cell_size(CELL_SIZE)

    return GRID_W, GRID_H, CELL_SIZE, SENSE_RADIUS, COMM_RANGE, OBSTACLES, ROBOTS_POS

def merge_maps(robots):
    """Merge all robots' local maps into a global map for visualization."""
    merged = None

    for r in robots:
        arr = np.asarray(r.map.explored_map)
        if merged is None:
            merged = arr.copy()
        else:
            mask = (arr != -1)  # only overwrite known cells
            merged[mask] = arr[mask]

    return merged

def setup_pygame(grid_w, grid_h, cell_size, label_h=25, fps=30, record=False):
    """Initialize Pygame window and optional video recording."""
    pygame.init()
    font = pygame.font.SysFont("Arial", 18, bold=True)

    screen_w = grid_w * cell_size
    screen_h = grid_h * cell_size + label_h
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Multi-Robot Exploration")
    clock = pygame.time.Clock()

    video = None
    if record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter("demo.mp4", fourcc, fps, (screen_w, screen_h))

    return screen, clock, font, video, fps

def render(screen, font, robots, merged, label_h):
    """Draw the global map and robot positions."""
    screen.fill((30, 30, 30))
    draw_label(screen, "Global Map", (10, 5), font)
    offset_global = (0, label_h)
    draw_grid(screen, merged, offset=offset_global)
    draw_robots(screen, robots, offset=offset_global)
    pygame.display.flip()

# -------------------- Main Loop --------------------
def main():
    # Load environment and robot configuration
    GRID_W, GRID_H, CELL_SIZE, SENSE_RADIUS, COMM_RANGE, OBSTACLES, ROBOTS_POS = load_config(map)

    # Initialize robots
    robots = [
        Robot(Map(GRID_W, GRID_H, OBSTACLES), pos, rid+1, SENSE_RADIUS, COMM_RANGE, strategy=strategy)
        for rid, pos in enumerate(ROBOTS_POS)
    ]
    for r in robots:
        r.robots = robots  # share reference to all robots

    # Animation & recording setup
    fps = 30

    if animation:
        screen, clock, font, video, fps = setup_pygame(GRID_W, GRID_H, CELL_SIZE, fps=fps, record=record)

    running = True
    steps = 0

    # Main exploration loop
    while running:
        running = False
        # Step each robot
        for r in robots:
            if r.step():
                running = True
        steps += 1

        # Merge local maps into global map
        merged = merge_maps(robots)

        # Render visualization
        if animation:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            render(screen, font, robots, merged, label_h=25)

            # Optional video recording
            if record and video is not None:
                frame = pygame.surfarray.array3d(screen)
                frame = np.transpose(frame, (1, 0, 2))
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video.write(frame_bgr)

            clock.tick(fps)

    # Finish recording
    if record and video is not None:
        for _ in range(int(fps * 1)):  # extra 1-second buffer
            video.write(frame_bgr)
        video.release()

    pygame.quit()

    # Compute completeness
    total_cells = merged.size
    unknown_cells = np.sum(merged == -1)
    completeness = 1.0 - (unknown_cells / total_cells)

    print("Steps:", steps)
    print("Completeness:", completeness)

    if strategy == "auction":
        for r in robots:
            print(f"Robot {r.id}: Auctions started={r.auctions_started}, kept={r.auctions_kept}, lost={r.auctions_lost_to}")

    return steps, completeness

# -------------------- Run --------------------
if __name__ == "__main__":
    main()