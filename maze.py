import heapq
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random


class MazeSolver:
    def __init__(self, maze):
        self.maze = maze
        self.rows = len(maze)
        self.cols = len(maze[0])

    @staticmethod
    def calculate_heuristic_cost(current, goal):
        # Using Manhattan distance as the heuristic
        return abs(goal[0] - current[0]) + abs(goal[1] - current[1])

    def is_valid(self, row, col):
        return 0 <= row < self.rows and 0 <= col < self.cols and self.maze[row][col] == 0

    def astar(self, start, goal):
        # A* algorithm implementation
        frontier = []  # Priority queue for frontier
        heapq.heappush(frontier, (0, start))  # (f(n), (row, col))

        came_from = {}  # Dictionary to store parent information
        cost_so_far = {}  # Dictionary to store the cost from start to each node
        came_from[start] = None
        cost_so_far[start] = 0

        while frontier:
            # loop through the maze 
            _, current = heapq.heappop(frontier)

            # if we find the goal we go back
            # from its came_from which is its parent to 
            # find the path
            if current == goal:
                path = []
                while current is not None:
                    path.insert(0, current)
                    current = came_from[current]
                return path

            row, col = current

            # loop through the each direction (right, left, up, down) and if it is valid
            # we calculate the new cost (each move has 1 cost) and then if new_cost < cost_so_far[(next_row, next_col)]
            # we add it to the queue and update the parent (came_from[(next_row, next_col)] = current)
            # and continue ......
            for (next_row, next_col) in [(row - 1, col), (row, col - 1), (row, col + 1), (row + 1, col)]:

                if self.is_valid(next_row, next_col):
                    new_cost = cost_so_far[current] + 1
                    if (next_row, next_col) not in cost_so_far or new_cost < cost_so_far[(next_row, next_col)]:
                        cost_so_far[(next_row, next_col)] = new_cost
                        priority = new_cost + self.calculate_heuristic_cost((next_row, next_col), goal)
                        heapq.heappush(frontier, (priority, (next_row, next_col)))
                        came_from[(next_row, next_col)] = current

        return None  # No path found

    def visualize_path(self, path):
        fig, ax = plt.subplots()

        # Plot the maze
        for row in range(self.rows):
            for col in range(self.cols):
                if self.maze[row][col] == 1:
                    rect = patches.Rectangle((col, self.rows - row - 1), 1, 1, linewidth=1, edgecolor='r', facecolor='r')
                    ax.add_patch(rect)

        # Plot the path
        for node in path:
            rect = patches.Rectangle((node[1], self.rows - node[0] - 1), 1, 1, linewidth=1, edgecolor='g', facecolor='g')
            ax.add_patch(rect)

        ax.set_xlim(0, self.cols)
        ax.set_ylim(0, self.rows)
        ax.set_aspect('equal', adjustable='box')

        plt.show()


def generate_maze_with_path():
    # Set a random size for the maze (you can adjust the range based on your preferences)
    rows = random.randint(5, 10)
    cols = random.randint(5, 10)

    # Generate a maze filled with walls (1s)
    maze = [[1] * cols for _ in range(rows)]

    # Set the start and end points
    start = (0, random.randint(0, cols - 1))
    end = (rows - 1, random.randint(0, cols - 1))

    # Create a path from start to end
    current = start
    while current != end:
        next_row = current[0] + random.choice([-1, 1])
        next_col = current[1] + random.choice([-1, 0, 1])

        if 0 <= next_row < rows and 0 <= next_col < cols:
            maze[next_row][next_col] = 0  # Mark the cell as a path
            current = (next_row, next_col)

    return {'maze': maze, 'start':start, 'end':end}
        

# Example Maze:
# maze 5 * 5
# maze = [
#     [0, 0, 0, 0, 0],
#     [1, 1, 1, 0, 1],
#     [0, 0, 0, 0, 0],
#     [1, 1, 0, 1, 1],
#     [0, 0, 0, 0, 0],
# ]

# maze 10 * 10
maze = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]
# result = generate_maze_with_path()
# maze = result['maze']

if maze:
    for row in maze:
        print(row)

solver = MazeSolver(maze)
start_node = (0, 0)
goal_node = (solver.rows - 1, solver.cols - 1)
# start_node = result['start']
# goal_node = result['end']

path = solver.astar(start_node, goal_node)

print('Start Point => ', start_node)
print('End Point => ', goal_node)
if path:
    print("Path found:", path)
    for way in path:
        x, y = way
        maze[x][y] = 2
    for row in maze:
        print(row)

    # graphics
    solver.visualize_path(path)

else:
    print("No path found.")


