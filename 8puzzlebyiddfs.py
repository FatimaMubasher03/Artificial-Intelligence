from collections import deque
GOAL_STATE = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

def find_tile(grid, value):
    for i, row in enumerate(grid):
        if value in row:
            return i, row.index(value)
    return -1, -1

MOVES = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}

def is_goal(state):
    return state == GOAL_STATE

def move_tile(state, direction):
    empty_row, empty_col = find_tile(state, 0)
    move_row, move_col = MOVES[direction]
    new_row, new_col = empty_row + move_row, empty_col + move_col
    if 0 <= new_row < 3 and 0 <= new_col < 3:
        new_state = [row[:] for row in state]  
        new_state[empty_row][empty_col], new_state[new_row][new_col] = (
            new_state[new_row][new_col], new_state[empty_row][empty_col])
        return new_state
    return None

def dfs_limited(state, depth, visited, path, limit):
    if is_goal(state):
        return path

    if depth >= limit:
        return None

    visited.add(tuple(map(tuple, state)))
    
    for direction in MOVES:
        new_state = move_tile(state, direction)
        if new_state and tuple(map(tuple, new_state)) not in visited:
            result = dfs_limited(new_state, depth + 1, visited, path + [direction], limit)
            if result:
                return result

    visited.remove(tuple(map(tuple, state)))
    return None

def iddfs(start_state, max_depth=50):
    explored_nodes_total = 0
    for depth in range(max_depth):
        visited = set()
        path = dfs_limited(start_state, 0, visited, [], depth)
        explored_nodes_total += len(visited)
        if path:
            return path, explored_nodes_total
    return None, explored_nodes_total

def print_solution(path):
    if path is None:
        print("No solution found.")
    else:
        print(f"Solution found with {len(path)} moves: {path}")

start_state = [[1, 5, 3], [2, 7, 4], [6, 0, 8]]  
goal_state = GOAL_STATE  

iddfs_solution, iddfs_nodes = iddfs(start_state)
print("IDDFS Solution:")
print_solution(iddfs_solution)
print(f"IDDFS explored {iddfs_nodes} nodes.")
