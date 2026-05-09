
import numpy as np
import heapq
import copy

class Problem:
    def __init__(self, initial_state):
        self.goal_state = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 0]
        ]   
        self.initial_state = initial_state
        self.total_expanded = 0
        self.max_queue_size = 1
        self.goal_depth = 0
    
    def search(self, heuristic):
        self.total_expanded = 0
        self.max_queue_size = 1
        self.goal_depth = 0
        myGraph = Graph(self.initial_state)
        while myGraph.frontier:
            node = myGraph.pop_frontier()

            if node.data == self.goal_state:
                self.goal_depth = node.depth
                self.print_solution(node, heuristic)
                return
            
            myGraph.mark_as_visited(node)
            self.total_expanded += 1
            for child in node.expand_node(): 
                if not myGraph.is_visited(child):
                    if heuristic == 1:
                        child.total_cost = child.path_cost
                    elif heuristic == 2:
                        child.total_cost = child.path_cost + child.calc_misplaced_tiles(self)
                    elif heuristic == 3:
                        child.total_cost = child.path_cost + child.calc_heuristic(self)

                    myGraph.add_to_frontier(child)
            
            self.max_queue_size = max(self.max_queue_size, len(myGraph.frontier))


   # prints backtracked solution for extra credit              
    def print_solution(self, node, heuristic_choice):
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
            print_puzzle(state, heuristic_choice, path_costs[iterator], euclidean_costs[iterator], misplaced_costs[iterator], is_root=is_root, is_goal=is_goal)
            iterator -= 1
            print()

        if heuristic_choice == 3 or heuristic_choice == 2 or heuristic_choice == 1:
            actions = list(reversed(actions))
            print("Sequence of actions to reach the goal:")
            for i, action in enumerate(actions, 1):
                print(f"Step {i}: {action}")

        print("Goal!!!")
        print(f"To solve this problem the search algorithm expanded a total of {self.total_expanded} nodes.")
        print(f"The maximum number of nodes in the queue at any one time: {self.max_queue_size}.")
        print(f"The depth of the goal node was {self.goal_depth}.")


class Node:
    def __init__(self, data, zero_position = (0, 1), parent=None, depth=0, path_cost=0, action=None):
        self.zero_position = zero_position
        self.parent = parent
        self.path_cost = path_cost
        self.total_cost = path_cost
        self.depth = depth
        self.action = action

        # data is the 2D matrix
        self.data = data
    
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

        for action, (dx, dy) in directions.items():
            new_row = self.zero_position[0] + dx
            new_col = self.zero_position[1] + dy
            if 0 <= new_row < len(self.data) and 0 <= new_col < len(self.data[0]):
                new_data = [row[:] for row in self.data]
                new_data[self.zero_position[0]][self.zero_position[1]], new_data[new_row][new_col] = (
                    new_data[new_row][new_col],
                    new_data[self.zero_position[0]][self.zero_position[1]]
                )
                children.append(Node(new_data, (new_row, new_col), self, depth, self.path_cost + 1, action))

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


class Graph:
    def __init__(self, initial_state):
        self.zero_pos = self.find_zero(initial_state)  
        self.root = Node(initial_state, self.zero_pos, action=None)
        self.visited = set()
        self.frontier = []  

        heapq.heappush(self.frontier, self.root)

    def find_zero(self, state):
        for row in range(len(state)):
            for column in range(len(state[0])):
                if state[row][column] == 0: 
                    return (row, column)
                
    def add_to_frontier(self, node):
        heapq.heappush(self.frontier, node)

    def pop_frontier(self):
        return heapq.heappop(self.frontier)

    
    def mark_as_visited(self, node):
        self.visited.add(tuple(map(tuple, node.data)))

    def is_visited(self, node):
        return tuple(map(tuple, node.data)) in self.visited



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



def get_user_puzzle():
    print("Enter your puzzle, use a zero to represent the blank:")
    puzzle = []
    for i in range(3):
        while True: 
            try:
                
                print(f"Enter row {i + 1} with a space between each number (e.g., '1 2 3')")
                row = list(map(int, input().split()))  
                
                if len(row) != 3:
                    raise ValueError("Invalid row length.")
    
                puzzle.append(row)

                break

            except ValueError:
                print("Follow instructions please")

            except KeyboardInterrupt:
                print("\n i guess you wanna leave.... k bye!!!")
                return

    return puzzle


def main():

    ###### TEST CASES ######


    print("EXAMPLE CASE IN REPORT DOC")
    reportDocNode = [
        [1, 2, 3],
        [4, 8, 0],
        [7, 6, 5]
    ]
    reportDoc = Problem(reportDocNode)
    reportDoc.search(1)
    reportDoc.search(2)
    reportDoc.search(3)


    print("TRIVAL")
    trivalNode = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 0]
    ]
    trival = Problem(trivalNode)
    trival.search(1)
    trival.search(2)
    trival.search(3)


    print("VERY EASY")
    veryEasyNode = [
        [1, 2, 0],
        [4, 5, 3],
        [7, 8, 6]
    ]
    veryEasy = Problem(veryEasyNode)
    veryEasy.search(1)
    veryEasy.search(2)
    veryEasy.search(3)


    print("EASY")
    easyNode = [
        [1, 2, 0],
        [4, 5, 3],
        [7, 8, 6]
    ]
    easy = Problem(easyNode)
    easy.search(1)
    easy.search(2)
    easy.search(3)


    print("DOABLE")
    doableNode = [
        [0, 1, 2],
        [4, 5, 3],
        [7, 8, 6]
    ]
    doable = Problem(doableNode)
    doable.search(1)
    doable.search(2)
    doable.search(3)


    print("OH BOY")
    ohBoyNode = [
        [8, 7, 1],
        [6, 0, 2],
        [5, 4, 3]
    ]
    ohBoy = Problem(ohBoyNode)
    ohBoy.search(1)
    ohBoy.search(2)
    ohBoy.search(3)


    # print("IMPOSSIBLE")
    # impossibleNode = [
    #     [1, 2, 3],
    #     [4, 5, 6],
    #     [8, 7, 0]
    # ]
    # impossible = Problem(impossibleNode)
    # impossible.search(1)
    # impossible.search(2)
    # impossible.search(3)


    ##### END OF TEST CASES #####

    data_choice = 0
    algo_choice = 0

    puzzle = [
        [1, 0, 3],
        [4, 2, 6],
        [7, 5, 8]
    ]

    print("Welcome to 8 puzzle solver")
    print("Type 1 to run example or 2 to enter your own")
    

    while True:
        try:
            data_choice = int(input())
            if data_choice in [1, 2]:
                break
            else:
                print("That's not an option")
        except ValueError:
            print("That's not an integer")
        except KeyboardInterrupt:
            print("\n welp i guess you wanna leave....bye!")
            return

    if data_choice == 2:
        puzzle = get_user_puzzle()

    problem = Problem(puzzle)

    print("Your puzzle is:")
    print_puzzle(puzzle, is_root=True)

    print("Enter your choice of algorithm")
    print("1 - Uniform cost search")
    print("2 - A* with misplaced tile heuristic")
    print("3 - A* with euclidean distance heuristic")
    

    while True:
        try:
            algo_choice = int(input())
            if algo_choice in [1, 2, 3]:
                break
            else:
                print("That's not an option")
        except ValueError:
            print("That's not and integer")
        except KeyboardInterrupt:
            print("\n welp i guess you wanna leave....bye!")
            return

    problem.search(algo_choice)

if __name__ == "__main__":
    main()