import pygame
from map import UNKNOWN, FREE, OCCUPIED
import random

CELL_SIZE = 25
COLOR_UNKNOWN = (130, 130, 130)
COLOR_FREE = (255, 255, 255)
COLOR_OCCUPIED = (0, 0, 0)

def id_to_color(robot_id):
    rng = random.Random(robot_id+5)
    return (
        rng.randint(50, 255),
        rng.randint(50, 255),
        rng.randint(50, 255)
    )

def draw_grid(surface, grid, offset=(0, 0)):
    """
    Draw only the occupancy grid.
    """
    ox, oy = offset
    h, w = grid.shape

    # Draw cells
    for y in range(h):
        for x in range(w):
            val = grid[y, x]
            if val == UNKNOWN:
                color = COLOR_UNKNOWN
            elif val == FREE:
                color = COLOR_FREE
            elif val == OCCUPIED:
                color = COLOR_OCCUPIED
            else:
                color = (255, 0, 255)  # unexpected

            pygame.draw.rect(
                surface, color,
                (ox + x * CELL_SIZE, oy + y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            )

    # Grid lines
    for x in range(w + 1):
        pygame.draw.line(surface, (200, 200, 200),
                         (ox + x * CELL_SIZE, oy),
                         (ox + x * CELL_SIZE, oy + h * CELL_SIZE))
    for y in range(h + 1):
        pygame.draw.line(surface, (200, 200, 200),
                         (ox, oy + y * CELL_SIZE),
                         (ox + w * CELL_SIZE, oy + y * CELL_SIZE))

def draw_robots(surface, robots, show_target=True, offset=(0, 0)):
    """
    Draw robots and their goals on top of the grid.
    """
    ox, oy = offset
    for r in robots:
        rx, ry = r.pos
        color = id_to_color(r.id)
        pygame.draw.circle(
            surface, color,
            (ox + rx * CELL_SIZE + CELL_SIZE // 2,
             oy + ry * CELL_SIZE + CELL_SIZE // 2),
            CELL_SIZE // 3
        )

        if getattr(r, 'target', None) is not None and show_target:
            gx, gy = r.target
            pygame.draw.rect(
                surface, color,
                (ox + gx * CELL_SIZE + CELL_SIZE // 4,
                 oy + gy * CELL_SIZE + CELL_SIZE // 4,
                 CELL_SIZE // 2, CELL_SIZE // 2),
                2
            )

def draw_label(surface, text, pos, font, color=(255, 255, 255)):
    label = font.render(text, True, color)
    surface.blit(label, pos)

def draw_links(surface, robots, offset=(0, 0)):
    """
    Draw communication links between robots and their neighbors.
    """
    ox, oy = offset
    for r in robots:
        rx, ry = r.pos
        x1 = ox + rx * CELL_SIZE + CELL_SIZE // 2
        y1 = oy + ry * CELL_SIZE + CELL_SIZE // 2
        for n in r.neighbors:
            nx, ny = n.pos
            x2 = ox + nx * CELL_SIZE + CELL_SIZE // 2
            y2 = oy + ny * CELL_SIZE + CELL_SIZE // 2
            pygame.draw.line(surface, (100, 200, 255), (x1, y1), (x2, y2), 2)
