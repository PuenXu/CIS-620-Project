import heapq

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

def astar(start, goal, explored_map):
    width = explored_map.shape[1]
    height = explored_map.shape[0]
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))
    came_from = {}
    cost_so_far = {start: 0}

    while open_set:
        _, cost, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            x, y = neighbor
            if 0 <= x < width and 0 <= y < height and explored_map[y, x] == 0:
                new_cost = cost_so_far[current] + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (priority, new_cost, neighbor))
                    came_from[neighbor] = current
                    
    return None  # No path found

def solve_tsp(start, frontier_samples):
    tour = []
    unvisited = list(frontier_samples)
    current = start

    while unvisited:
        next_point = min(unvisited, key=lambda p: heuristic(current, p))
        tour.append(next_point)
        unvisited.remove(next_point)
        current = next_point

    return tour