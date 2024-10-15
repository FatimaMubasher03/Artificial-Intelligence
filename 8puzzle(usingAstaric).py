import heapq

class PuzzleNode:
    def __init__(self, state, parent=None, move=None, g_cost=0, h_cost=0):
        self.state = state 
        self.parent = parent  
        self.move = move  
        self.g_cost = g_cost
        self.h_cost = h_cost  
        self.f_cost = g_cost + h_cost  

    def __lt__(self, other):
        return self.f_cost < other.f_cost

    def generate_children(self):
        children = []
        x, y = self.find_empty_tile()
        directions = {'up': (x - 1, y), 'down': (x + 1, y), 'left': (x, y - 1), 'right': (x, y + 1)}
        
        for move, (new_x, new_y) in directions.items():
            if 0 <= new_x < 3 and 0 <= new_y < 3:
                new_state = [row[:] for row in self.state]  
                new_state[x][y], new_state[new_x][new_y] = new_state[new_x][new_y], new_state[x][y]  
                children.append(PuzzleNode(new_state, self, move, self.g_cost + 1))
        
        return children

    def find_empty_tile(self):
        for i in range(3):
            for j in range(3):
                if self.state[i][j] == 0:
                    return i, j

    def calculate_heuristic(self, goal_state, heuristic='manhattan'):
        if heuristic == 'manhattan':
            return self.manhattan_distance(goal_state)
        elif heuristic == 'misplaced':
            return self.misplaced_tiles(goal_state)

    def manhattan_distance(self, goal_state):
        distance = 0
        for i in range(3):
            for j in range(3):
                value = self.state[i][j]
                if value != 0:
                    goal_x, goal_y = divmod(goal_state.index(value), 3)
                    distance += abs(i - goal_x) + abs(j - goal_y)
        return distance

    def misplaced_tiles(self, goal_state):
        misplaced = 0
        for i in range(3):
            for j in range(3):
                if self.state[i][j] != 0 and self.state[i][j] != goal_state[i * 3 + j]:
                    misplaced += 1
        return misplaced

class AStarSolver:
    def __init__(self, start_state, goal_state):
        self.start_state = start_state
        self.goal_state = goal_state
        self.goal_state_flat = sum(goal_state, [])
        
    def solve(self, heuristic='manhattan'):
        open_list = []
        closed_set = set()

        start_node = PuzzleNode(self.start_state)
        start_node.h_cost = start_node.calculate_heuristic(self.goal_state_flat, heuristic)
        start_node.f_cost = start_node.g_cost + start_node.h_cost


        heapq.heappush(open_list, start_node)

        while open_list:

            current_node = heapq.heappop(open_list)

            if current_node.state == self.goal_state:
                return self.trace_solution(current_node)

   
            closed_set.add(tuple(map(tuple, current_node.state)))

            for child in current_node.generate_children():
                if tuple(map(tuple, child.state)) in closed_set:
                    continue  

                child.h_cost = child.calculate_heuristic(self.goal_state_flat, heuristic)
                child.f_cost = child.g_cost + child.h_cost

                heapq.heappush(open_list, child)

        return None 

    def trace_solution(self, node):
        path = []
        while node:
            path.append(node.state)
            node = node.parent
        return path[::-1] 

    def is_solvable(self, state):
        flat_state = sum(state, [])
        inversions = 0
        for i in range(len(flat_state)):
            for j in range(i + 1, len(flat_state)):
                if flat_state[i] != 0 and flat_state[j] != 0 and flat_state[i] > flat_state[j]:
                    inversions += 1
        return inversions % 2 == 0

def print_solution(solution):
    if solution:
        for state in solution:
            for row in state:
                print(row)
            print()
    else:
        print("No solution found.")
        

start_state = [
    [1, 2, 3],
    [4, 0, 5],
    [7, 8, 6]
]

goal_state = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
]

solver = AStarSolver(start_state, goal_state)
if solver.is_solvable(start_state):
    solution = solver.solve(heuristic='manhattan')
    print_solution(solution)
else:
    print("The puzzle is not solvable.")
