import pygame
import numpy as np
import random
import json

# Map setup constants
CELL_SIZE = 25
# GRID_W, GRID_H = 20, 15
GRID_W, GRID_H = 30, 25
# GRID_W, GRID_H = 40, 35

COLOR_UNKNOWN = (130, 130, 130)
COLOR_FREE = (255, 255, 255)
COLOR_OCCUPIED = (0, 0, 0)
GRID_COLOR = (200, 200, 200)

# Robot Setup
SENSE_RADIUS = 2
COMM_RANGE = 6

def id_to_color(robot_id):
    rng = random.Random(robot_id+5)
    return (
        rng.randint(50, 255),
        rng.randint(50, 255),
        rng.randint(50, 255)
    )

def select_obstacles():
    pygame.init()
    screen = pygame.display.set_mode((GRID_W * CELL_SIZE, GRID_H * CELL_SIZE))
    pygame.display.set_caption("Map Editor - Click or Drag to Toggle Obstacles")

    grid = np.zeros((GRID_H, GRID_W), dtype=int)

    running = True
    mouse_down = False
    draw_mode = None

    while running:
        screen.fill(COLOR_FREE)
        for y in range(GRID_H):
            for x in range(GRID_W):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if grid[y, x] == 1:
                    pygame.draw.rect(screen, COLOR_OCCUPIED, rect)
                else:
                    pygame.draw.rect(screen, COLOR_FREE, rect)
                pygame.draw.rect(screen, GRID_COLOR, rect, 1)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                gx, gy = mx // CELL_SIZE, my // CELL_SIZE

                mouse_down = True
                if event.button == 1:
                    draw_mode = 1
                elif event.button == 3:
                    draw_mode = 0

                grid[gy, gx] = draw_mode

            elif event.type == pygame.MOUSEBUTTONUP:
                mouse_down = False
                draw_mode = None

            elif event.type == pygame.MOUSEMOTION and mouse_down:
                mx, my = pygame.mouse.get_pos()
                gx, gy = mx // CELL_SIZE, my // CELL_SIZE
                if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
                    grid[gy, gx] = draw_mode

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    running = False

    pygame.quit()
    return [(x, y) for y in range(GRID_H) for x in range(GRID_W) if grid[y, x] == 1]

def select_robots(obstacles):
    pygame.init()
    screen = pygame.display.set_mode((GRID_W * CELL_SIZE, GRID_H * CELL_SIZE))
    pygame.display.set_caption("Map Editor - Select Robot Positions")

    grid = np.zeros((GRID_H, GRID_W), dtype=int)
    for ox, oy in obstacles:
        grid[oy, ox] = 1

    robot_positions = []
    running = True

    while running:
        screen.fill(COLOR_FREE)
        for y in range(GRID_H):
            for x in range(GRID_W):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if grid[y, x] == 1:
                    pygame.draw.rect(screen, COLOR_OCCUPIED, rect)
                elif (x, y) in [pos for pos, _ in robot_positions]:
                    robot_id = [rid for pos, rid in robot_positions if pos == (x, y)][0]
                    pygame.draw.rect(screen, id_to_color(robot_id), rect)
                else:
                    pygame.draw.rect(screen, COLOR_FREE, rect)
                pygame.draw.rect(screen, GRID_COLOR, rect, 1)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = pygame.mouse.get_pos()
                gx, gy = mx // CELL_SIZE, my // CELL_SIZE
                if (gx, gy) not in obstacles and (gx, gy) not in [p for p, _ in robot_positions]:
                    robot_positions.append(((gx, gy), len(robot_positions) + 1))

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                running = False

    pygame.quit()
    return [pos for pos, _ in robot_positions]

def randomize_robots(obstacles, num_robots):
    free_cells = [(x, y) for y in range(GRID_H) for x in range(GRID_W) if (x, y) not in obstacles]
    
    if num_robots > len(free_cells):
        raise ValueError("Not enough free cells to place all robots!")
    
    return random.sample(free_cells, num_robots)

def save_config(obstacles, robots):
    config = {
        "cell_size": CELL_SIZE,
        "grid_w": GRID_W,
        "grid_h": GRID_H,
        "sense_radius": SENSE_RADIUS,
        "comm_range": COMM_RANGE,
        "obstacles": [tuple(pos) for pos in obstacles], 
        "robots": [tuple(pos) for pos in robots]       
    }

    with open("maps/custom_map.json", "w") as f:
        json.dump(config, f, indent=4)
    print("Configuration saved to custom_map.json")

def visualize_map(json_file="maps/custom_map.json"):
    # Load map config
    with open(json_file, "r") as f:
        cfg = json.load(f)

    GRID_W = cfg["grid_w"]
    GRID_H = cfg["grid_h"]
    CELL_SIZE = cfg["cell_size"]
    OBSTACLES = [tuple(pos) for pos in cfg["obstacles"]]
    ROBOTS_POS = [tuple(pos) for pos in cfg["robots"]]

    pygame.init()
    screen = pygame.display.set_mode((GRID_W * CELL_SIZE, GRID_H * CELL_SIZE))
    pygame.display.set_caption("Map Verification")

    running = True
    clock = pygame.time.Clock()

    while running:
        screen.fill(COLOR_FREE)

        # Draw grid & obstacles
        for y in range(GRID_H):
            for x in range(GRID_W):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if (x, y) in OBSTACLES:
                    pygame.draw.rect(screen, COLOR_OCCUPIED, rect)
                else:
                    pygame.draw.rect(screen, COLOR_FREE, rect)
                pygame.draw.rect(screen, GRID_COLOR, rect, 1)

        # Draw robots
        for rid, (rx, ry) in enumerate(ROBOTS_POS, start=1):
            color = id_to_color(rid)
            pygame.draw.circle(
                screen, color,
                (rx * CELL_SIZE + CELL_SIZE // 2, ry * CELL_SIZE + CELL_SIZE // 2),
                CELL_SIZE // 3
            )

        pygame.display.flip()
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False

    pygame.quit()


if __name__ == "__main__":
    OBSTACLES = select_obstacles()

    ROBOTS_POS = select_robots(OBSTACLES)

    save_config(OBSTACLES, ROBOTS_POS)

    visualize_map("maps/custom_map.json")