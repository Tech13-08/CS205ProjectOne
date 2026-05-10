import numpy as np
import heapq
import copy

class Problem:
    def __init__(self, initial_state):
        self.goal_state = [
            [-1, -1, -1,  0, -1,  0, -1,  0, -1, -1],
            [ 1,  2,  3,  4,  5,  6,  7,  8,  9,  0]
        ]   
        self.initial_state = initial_state

   # prints backtracked solution for extra credit              
    def print_solution(self, node, queueing_function, total_expanded=0, max_queue_size=0, goal_depth=0):
        path = []
        actions = []
        path_costs = []
        euclidean_costs = []
        misplaced_costs = []
        
        while node:
            path_costs.append(node.path_cost)
            euclidean_costs.append(node.calc_heuristic(self)) 
            misplaced_costs.append(node.calc_misplaced_tiles(self)) 
            path.append(node.data)
            if node.action:
                actions.append(node.action)
            node = node.parent
        
        iterator = len(path_costs) - 1
        # Mark the root node and the goal node
        for idx, state in enumerate(reversed(path)):
            is_root = idx == 0
            is_goal = iterator == 0
            print_puzzle(state, queueing_function, path_costs[iterator], euclidean_costs[iterator], misplaced_costs[iterator], is_root=is_root, is_goal=is_goal)
            iterator -= 1
            print()

        if queueing_function == 3 or queueing_function == 2 or queueing_function == 1:
            actions = list(reversed(actions))
            print("Sequence of actions to reach the goal:")
            for i, action in enumerate(actions, 1):
                print(f"Step {i}: {action}")

        print("Goal!!!")
        print(f"To solve this problem the search algorithm expanded a total of {total_expanded} nodes.")
        print(f"The maximum number of nodes in the queue at any one time: {max_queue_size}.")
        print(f"The depth of the goal node was {goal_depth}.")


class Node:
    def __init__(self, data, parent=None, depth=0, path_cost=0, action=None):
        self.parent = parent
        self.path_cost = path_cost
        self.total_cost = path_cost
        self.depth = depth
        self.action = action

        # data is the 2D matrix
        self.data = data

        self.zero_position = self._find_zero()

    def _find_zero(self):
        zeros = []
        for r in range(len(self.data)):
            for c in range(len(self.data[0])):
                if self.data[r][c] == 0:
                    zeros.append((r, c))
        return zeros
    
    def __lt__(self, other):
        if self.total_cost == other.total_cost:
            return self.zero_position[0] < other.zero_position[0] or self.zero_position[1] < other.zero_position[1]
        return self.total_cost < other.total_cost

    def expand_node(self):
        depth = self.depth + 1
        children = []
        directions = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1)
        }

        for row, column in self._find_zeros():
            for action, (dx, dy) in directions.items():
                new_row = row + dx
                new_col = column + dy
                if 0 <= new_row < len(self.data) and 0 <= new_col < len(self.data[0]):
                    if self.data[new_row][new_col] == -1:
                        continue
                    new_data = [row[:] for row in self.data]
                    new_data[row][column], new_data[new_row][new_col] = (
                        new_data[new_row][new_col],
                        new_data[row][column]
                    )
                    children.append(Node(new_data, self, depth, self.path_cost + 1, action))

        return children
    
    def calc_heuristic(self, problem):
        h = 0
        ##### can be optimized to have two sets of double for loops while keeping track of a list of coords #####
        ##### i would change this for if we expand past 3x3 grids #####
        for row in range(len(problem.goal_state)):
            for column in range(len(problem.goal_state[0])):
                # inside checker that finds other values position
                if self.data[row][column] != problem.goal_state[row][column]:
                    for search_row in range(len(problem.goal_state)):
                        for search_column in range(len(problem.goal_state[0])):
                            if self.data[row][column] == problem.goal_state[search_row][search_column]:
                                h += (np.sqrt((row - search_row)**2 + (column - search_column)**2))
        return h
    
    def calc_misplaced_tiles(self, problem):
        h = 0
        for row in range(len(problem.goal_state)):
            for column in range(len(problem.goal_state[0])):
                if self.data[row][column] != problem.goal_state[row][column] and self.data[row][column] != 0:
                    h += 1
        return h


def general_search(problem, queueing_function):
    total_expanded = 0
    max_queue_size = 1
    visited = set()
    nodes = [Node(problem.initial_state)]
    while nodes:
        node = heapq.heappop(nodes)

        if node.data == problem.goal_state:
            problem.goal_depth = node.depth
            problem.print_solution(node, queueing_function, total_expanded, max_queue_size, problem.goal_depth)
            return
        
        state = tuple(map(tuple, node.data))
        if state not in visited:
            visited.add(state)
            total_expanded += 1
            for child in node.expand_node(): 
                if queueing_function == 1:
                    child.total_cost = child.path_cost
                elif queueing_function == 2:
                    child.total_cost = child.path_cost + child.calc_misplaced_tiles(problem)
                elif queueing_function == 3:
                    child.total_cost = child.path_cost + child.calc_heuristic(problem)

                heapq.heappush(nodes, child)
        
        max_queue_size = max(max_queue_size, len(nodes))


# menu functions

def print_puzzle(state, heuristic_choice=1, path_cost=0, euclidean_cost=0, misplaced_cost=0, is_root=False, is_goal=False):
    if is_root:
        print(f"This is the initial state:")
    elif is_goal:
        print(f"This is the goal state:")
    else:
        if heuristic_choice == 1:
            print(f"The best state to expand with g(n) = {float(path_cost)} is...")
        elif heuristic_choice == 2:
            print(f"The best state to expand with g(n) = {float(path_cost)} and h(n) = {round(float(misplaced_cost), 3)} is...")
        elif heuristic_choice == 3:
            print(f"The best state to expand with g(n) = {float(path_cost)} and h(n) = {round(float(euclidean_cost), 3)} is...")
    
    for row in state:
        print(' '.join(str(x) for x in row))



def main():
    puzzle = [
        [-1, -1, -1,  0, -1,  0, -1,  0, -1, -1],
    [ 1,  2,  3,  4,  5,  6,  7,  8,  0,  9]
    ]
    print("Enter your choice of algorithm")
    print("1 - Uniform cost search")
    print("2 - A* with misplaced tile heuristic")
    print("3 - A* with euclidean distance heuristic")
    choice = int(input())
    
    prob = Problem(puzzle)
    general_search(prob, choice)

if __name__ == "__main__":
    main()