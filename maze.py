import matplotlib.pyplot as plt


plt.set_loglevel (level = 'warning')


class Cell(object):
    """Class for representing a cell in a 2D grid.

        Attributes:
            row (int): The row that this cell belongs to
            col (int): The column that this cell belongs to
            visited (bool): True if this cell has been visited by an algorithm
            active (bool):
            is_entry_exit (bool): True when the cell is the beginning or end of the maze
            walls (list):
            neighbours (list):
    """
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.visited = False
        self.active = False
        self.is_entry_exit = None
        self.walls = {"top": True, "right": True, "bottom": True, "left": True}
        self.neighbours = list()

    def is_walls_between(self, neighbour):
        """Function that checks if there are walls between self and a neighbour cell.
        Returns true if there are walls between. Otherwise returns False.

        Args:
            neighbour The cell to check between

        Return:
            True: If there are walls in between self and neighbor
            False: If there are no walls in between the neighbors and self

        """
        if self.row - neighbour.row == 1 and self.walls["top"] and neighbour.walls["bottom"]:
            return True
        elif self.row - neighbour.row == -1 and self.walls["bottom"] and neighbour.walls["top"]:
            return True
        elif self.col - neighbour.col == 1 and self.walls["left"] and neighbour.walls["right"]:
            return True
        elif self.col - neighbour.col == -1 and self.walls["right"] and neighbour.walls["left"]:
            return True

        return False

    def remove_walls(self, neighbour_row, neighbour_col):
        """Function that removes walls between neighbour cell given by indices in grid.

            Args:
                neighbour_row (int):
                neighbour_col (int):

            Return:
                True: If the operation was a success
                False: If the operation failed

        """
        if self.row - neighbour_row == 1:
            self.walls["top"] = False
            return True, ""
        elif self.row - neighbour_row == -1:
            self.walls["bottom"] = False
            return True, ""
        elif self.col - neighbour_col == 1:
            self.walls["left"] = False
            return True, ""
        elif self.col - neighbour_col == -1:
            self.walls["right"] = False
            return True, ""
        return False

    def set_as_entry_exit(self, entry_exit, row_limit, col_limit):
        """Function that sets the cell as an entry/exit cell by
        disabling the outer boundary wall.
        First, we check if the entrance/exit is on the top row. Next, we check if it should
        be on the bottom row. Finally, we check if it is on the left wall or the bottom row.

        Args:
            entry_exit: True to set this cell as an exit/entry. False to remove it as one
            row_limit:
            col_limit:
        """

        if self.row == 0:
            self.walls["top"] = False
        elif self.row == row_limit:
            self.walls["bottom"] = False
        elif self.col == 0:
            self.walls["left"] = False
        elif self.col == col_limit:
            self.walls["right"] = False

        self.is_entry_exit = entry_exit


import time
import random
import math

# global variable to store list of all available algorithms
algorithm_list = ["dfs_backtrack", "bin_tree"]

def depth_first_recursive_backtracker( maze, start_coor ):
        k_curr, l_curr = start_coor             # Where to start generating
        path = [(k_curr, l_curr)]               # To track path of solution
        maze.grid[k_curr][l_curr].visited = True     # Set initial cell to visited
        visit_counter = 1                       # To count number of visited cells
        visited_cells = list()                  # Stack of visited cells for backtracking

        # print("\nGenerating the maze with depth-first search...")
        time_start = time.time()

        while visit_counter < maze.grid_size:     # While there are unvisited cells
            neighbour_indices = maze.find_neighbours(k_curr, l_curr)    # Find neighbour indicies
            neighbour_indices = maze._validate_neighbours_generate(neighbour_indices)

            if neighbour_indices is not None:   # If there are unvisited neighbour cells
                visited_cells.append((k_curr, l_curr))              # Add current cell to stack
                k_next, l_next = random.choice(neighbour_indices)     # Choose random neighbour
                maze.grid[k_curr][l_curr].remove_walls(k_next, l_next)   # Remove walls between neighbours
                maze.grid[k_next][l_next].remove_walls(k_curr, l_curr)   # Remove walls between neighbours
                maze.grid[k_next][l_next].visited = True                 # Move to that neighbour
                k_curr = k_next
                l_curr = l_next
                path.append((k_curr, l_curr))   # Add coordinates to part of generation path
                visit_counter += 1

            elif len(visited_cells) > 0:  # If there are no unvisited neighbour cells
                k_curr, l_curr = visited_cells.pop()      # Pop previous visited cell (backtracking)
                path.append((k_curr, l_curr))   # Add coordinates to part of generation path

        # print("Number of moves performed: {}".format(len(path)))
        # print("Execution time for algorithm: {:.4f}".format(time.time() - time_start))

        # maze.grid[maze.entry_coor[0]][maze.entry_coor[1]].set_as_entry_exit("entry",
        #     maze.num_rows-1, maze.num_cols-1)
        # for i in range(len(maze.exit_coor)):
        #     maze.grid[maze.exit_coor[i][0]][maze.exit_coor[i][1]].set_as_entry_exit("exit",
        #         maze.num_rows-1, maze.num_cols-1)

        for i in range(maze.num_rows):
            for j in range(maze.num_cols):
                maze.grid[i][j].visited = False      # Set all cells to unvisited before returning grid

        maze.generation_path = path

def binary_tree( maze, start_coor ):
    # store the current time
    time_start = time.time()

    # repeat the following for all rows
    for i in range(0, maze.num_rows):

        # check if we are in top row
        if( i == maze.num_rows - 1 ):
            # remove the right wall for this, because we cant remove top wall
            for j in range(0, maze.num_cols-1):
                maze.grid[i][j].remove_walls(i, j+1)
                maze.grid[i][j+1].remove_walls(i, j)

            # go to the next row
            break

        # repeat the following for all cells in rows
        for j in range(0, maze.num_cols):

            # check if we are in the last column
            if( j == maze.num_cols-1 ):
                # remove only the top wall for this cell
                maze.grid[i][j].remove_walls(i+1, j)
                maze.grid[i+1][j].remove_walls(i, j)
                continue

            # for all other cells
            # randomly choose between 0 and 1.
            # if we get 0, remove top wall; otherwise remove right wall
            remove_top = random.choice([True,False])

            # if we chose to remove top wall
            if remove_top:
                maze.grid[i][j].remove_walls(i+1, j)
                maze.grid[i+1][j].remove_walls(i, j)
            # if we chose top remove right wall
            else:
                maze.grid[i][j].remove_walls(i, j+1)
                maze.grid[i][j+1].remove_walls(i, j)

    # print("Number of moves performed: {}".format(maze.num_cols * maze.num_rows))
    # print("Execution time for algorithm: {:.4f}".format(time.time() - time_start))

    # choose the entry and exit coordinates
    # maze.grid[maze.entry_coor[0]][maze.entry_coor[1]].set_as_entry_exit("entry",
    #     maze.num_rows-1, maze.num_cols-1)
    for i in range(len(maze.exit_coor)):
        maze.grid[maze.exit_coor[i][0]][maze.exit_coor[i][1]].set_as_entry_exit("exit",
            maze.num_rows-1, maze.num_cols-1)

    # create a path for animating the maze creation using a binary tree
    path = list()
    # variable for holding number of cells visited until now
    visit_counter = 0
    # created list of cell visited uptil now to for backtracking
    visited = list()

    # create variables to hold the coords of current cell
    # no matter what the user gives as start coords, we choose the
    k_curr, l_curr = (maze.num_rows-1, maze.num_cols-1)
    # add first cell to the path
    path.append( (k_curr,l_curr) )

    # mark first cell as visited
    begin_time = time.time()

    # repeat until all the cells have been visited
    while visit_counter < maze.grid_size:     # While there are unvisited cells

        # for each cell, we only visit top and right cells.
        possible_neighbours = list()

        try:
            # take only those cells that are unvisited and accessible
            if not maze.grid[k_curr-1][l_curr].visited and k_curr != 0:
                if not maze.grid[k_curr][l_curr].is_walls_between(maze.grid[k_curr-1][l_curr]):
                    possible_neighbours.append( (k_curr-1,l_curr))
        except:
            print()

        try:
            # take only those cells that are unvisited and accessible
            if not maze.grid[k_curr][l_curr-1].visited and l_curr != 0:
                if not maze.grid[k_curr][l_curr].is_walls_between(maze.grid[k_curr][l_curr-1]):
                    possible_neighbours.append( (k_curr,l_curr-1))
        except:
            print()

        # if there are still traversible cell from current cell
        if len( possible_neighbours ) != 0:
            # select to first element to traverse
            k_next, l_next = possible_neighbours[0]
            # add this cell to the path
            path.append( possible_neighbours[0] )
            # add this cell to the visited
            visited.append( (k_curr,l_curr) )
            # mark this cell as visited
            maze.grid[k_next][l_next].visited = True

            visit_counter+= 1

            # update the current cell coords
            k_curr, l_curr = k_next, l_next

        else:
            # check if no more cells can be visited
            if len( visited ) != 0:
                k_curr, l_curr = visited.pop()
                path.append( (k_curr,l_curr) )
            else:
                break
    for row in maze.grid:
        for cell in row:
            cell.visited = False

    # print(f"Generating path for maze took {time.time() - begin_time}s.")
    maze.generation_path = path



import random
import math
import time


class Maze(object):
    """Class representing a maze; a 2D grid of Cell objects. Contains functions
    for generating randomly generating the maze as well as for solving the maze.

    Attributes:
        num_cols (int): The height of the maze, in Cells
        num_rows (int): The width of the maze, in Cells
        id (int): A unique identifier for the maze
        grid_size (int): The area of the maze, also the total number of Cells in the maze
        entry_coor Entry location cell of maze
        exit_coor Exit location cell of maze
        generation_path : The path that was taken when generating the maze
        solution_path : The path that was taken by a solver when solving the maze
        initial_grid (list):
        grid (list): A copy of initial_grid (possible this is un-needed)
        """

    def __init__(self, num_rows, num_cols, id=0, algorithm = "dfs_backtrack"):
        """Creates a gird of Cell objects that are neighbors to each other.

            Args:
                    num_rows (int): The width of the maze, in cells
                    num_cols (int): The height of the maze in cells
                    id (id): An unique identifier

        """
        self.num_cols = num_cols
        self.num_rows = num_rows
        self.id = id
        self.grid_size = num_rows*num_cols
        self.num_exits = random.randint(1, (num_cols+num_rows)*2+1)
        self.entry_coor = (0,0)
        self.generation_path = []
        self.solution_path = None
        self.initial_grid = self.generate_grid()
        self.grid = self.initial_grid
        self.generate_maze(algorithm, (0, 0))
        self.add_padding()
        self.possible_exits = [((0, i), (1, i)) for i in range(1, self.num_cols-1)]+[((self.num_rows-1, i), (self.num_rows-2, i)) for i in range(1, self.num_cols-1)]+\
            [((i, 0), (i, 1)) for i in range(1, self.num_rows-1)]+[((i, self.num_cols-1), (i, self.num_cols-2)) for i in range(1, self.num_rows-1)]
        self.exits = random.choices(self.possible_exits, k=self.num_exits)
        # self.make_exits()

    def generate_grid(self):
        """Function that creates a 2D grid of Cell objects. This can be thought of as a
        maze without any paths carved out

        Return:
            A list with Cell objects at each position

        """

        # Create an empty list
        grid = list()

        # Place a Cell object at each location in the grid
        for i in range(self.num_rows):
            grid.append(list())

            for j in range(self.num_cols):
                grid[i].append(Cell(i, j))

        return grid

    def find_neighbours(self, cell_row, cell_col):
        """Finds all existing and unvisited neighbours of a cell in the
        grid. Return a list of tuples containing indices for the unvisited neighbours.

        Args:
            cell_row (int):
            cell_col (int):

        Return:
            None: If there are no unvisited neighbors
            list: A list of neighbors that have not been visited
        """
        neighbours = list()

        def check_neighbour(row, col):
            # Check that a neighbour exists and that it's not visited before.
            if row >= 0 and row < self.num_rows and col >= 0 and col < self.num_cols:
                neighbours.append((row, col))

        check_neighbour(cell_row-1, cell_col)     # Top neighbour
        check_neighbour(cell_row, cell_col+1)     # Right neighbour
        check_neighbour(cell_row+1, cell_col)     # Bottom neighbour
        check_neighbour(cell_row, cell_col-1)     # Left neighbour

        if len(neighbours) > 0:
            return neighbours

        else:
            return None     # None if no unvisited neighbours found

    def _validate_neighbours_generate(self, neighbour_indices):
        """Function that validates whether a neighbour is unvisited or not. When generating
        the maze, we only want to move to move to unvisited cells (unless we are backtracking).

        Args:
            neighbour_indices:

        Return:
            True: If the neighbor has been visited
            False: If the neighbor has not been visited

        """

        neigh_list = [n for n in neighbour_indices if not self.grid[n[0]][n[1]].visited]

        if len(neigh_list) > 0:
            return neigh_list
        else:
            return None

    def validate_neighbours_solve(self, neighbour_indices, k, l, k_end, l_end, method = "fancy"):
        """Function that validates whether a neighbour is unvisited or not and discards the
        neighbours that are inaccessible due to walls between them and the current cell. The
        function implements two methods for choosing next cell; one is 'brute-force' where one
        of the neighbours are chosen randomly. The other is 'fancy' where the next cell is chosen
        based on which neighbour that gives the shortest distance to the final destination.

        Args:
            neighbour_indices
            k
            l
            k_end
            l_end
            method

        Return:


        """
        if method == "fancy":
            neigh_list = list()
            min_dist_to_target = 100000

            for k_n, l_n in neighbour_indices:
                if (not self.grid[k_n][l_n].visited
                        and not self.grid[k][l].is_walls_between(self.grid[k_n][l_n])):
                    dist_to_target = math.sqrt((k_n - k_end) ** 2 + (l_n - l_end) ** 2)

                    if (dist_to_target < min_dist_to_target):
                        min_dist_to_target = dist_to_target
                        min_neigh = (k_n, l_n)

            if "min_neigh" in locals():
                neigh_list.append(min_neigh)

        elif method == "brute-force":
            neigh_list = [n for n in neighbour_indices if not self.grid[n[0]][n[1]].visited
                          and not self.grid[k][l].is_walls_between(self.grid[n[0]][n[1]])]

        if len(neigh_list) > 0:
            return neigh_list
        else:
            return None

    def _pick_random_entry_exit(self, used_entry_exit=None):
        """Function that picks random coordinates along the maze boundary to represent either
        the entry or exit point of the maze. Makes sure they are not at the same place.

        Args:
            used_entry_exit

        Return:

        """
        possible_exits = [(0, i) for i in range(self.num_cols)] + [(self.num_rows)]
        rng_side = random.randint(0, 3)
        rng_entry_exit = None
        if (rng_side == 0):     # Top side
            rng_entry_exit = (0, random.randint(0, self.num_cols-1))

        elif (rng_side == 2):   # Right side
            rng_entry_exit = (self.num_rows-1, random.randint(0, self.num_cols-1))

        elif (rng_side == 1):   # Bottom side
            rng_entry_exit = (random.randint(0, self.num_rows-1), self.num_cols-1)

        elif (rng_side == 3):   # Left side
            rng_entry_exit = (random.randint(0, self.num_rows-1), 0)    # Initialize with used value

        # Try until unused location along boundary is found.
        while rng_entry_exit in used_entry_exit:
            rng_side = random.randint(0, 3)

            if (rng_side == 0):     # Top side
                rng_entry_exit = (0, random.randint(0, self.num_cols-1))

            elif (rng_side == 2):   # Right side
                rng_entry_exit = (self.num_rows-1, random.randint(0, self.num_cols-1))

            elif (rng_side == 1):   # Bottom side
                rng_entry_exit = (random.randint(0, self.num_rows-1), self.num_cols-1)

            elif (rng_side == 3):   # Left side
                rng_entry_exit = (random.randint(0, self.num_rows-1), 0)
        used_entry_exit.add(rng_entry_exit)
        return used_entry_exit       # Return entry/exit that is different from exit/entry

    def generate_maze(self, algorithm, start_coor = (0, 0)):
        """This takes the internal grid object and removes walls between cells using the
        depth-first recursive backtracker algorithm.

        Args:
            start_coor: The starting point for the algorithm

        """

        if algorithm == "dfs_backtrack":
            depth_first_recursive_backtracker(self, start_coor)
        elif algorithm == "bin_tree":
            binary_tree(self, start_coor)
    def add_padding(self):
        top_grid_walls= [False] + [self.initial_grid[0][i].walls['top'] for i in range(len(self.initial_grid[0]))] + [False]
        top_grid = [Cell(0, i) for i in range(len(self.initial_grid[0]) + 2)]
        for i in range(len(top_grid)):
            top_grid[i].walls['top'] = False
            top_grid[i].walls['bottom'] = top_grid_walls[i]
            top_grid[i].walls['left'] = False
            top_grid[i].walls['right'] = False
        left_grid_walls = [self.initial_grid[i][0].walls['left'] for i in range(len(self.initial_grid))] + [False]
        right_grid_walls = [self.initial_grid[i][-1].walls['right'] for i in range(len(self.initial_grid))] + [False]
        bottom_grid_walls = [self.initial_grid[-1][i].walls['bottom'] for i in range(len(self.initial_grid[-1]))]
        left_grid = [Cell(i, 0) for i in range(1, len(self.initial_grid) + 2)]
        right_grid = [Cell(i, len(self.initial_grid)) for i in range(1, len(self.initial_grid) + 2)]
        bottom_grid = [Cell(len(self.initial_grid[-1]), i) for i in range(1, len(self.initial_grid[-1]) + 1)]
        for i in range(len(left_grid)):
            left_grid[i].walls['top'] = False
            left_grid[i].walls['bottom'] = False
            left_grid[i].walls['left'] = False
            left_grid[i].walls['right'] = left_grid_walls[i]
        for i in range(len(right_grid)):
            right_grid[i].walls['top'] = False
            right_grid[i].walls['bottom'] = False
            right_grid[i].walls['left'] = right_grid_walls[i]
            right_grid[i].walls['right'] = False
        for i in range(len(bottom_grid)):
            bottom_grid[i].walls['top'] = bottom_grid_walls[i]
            bottom_grid[i].walls['bottom'] = False
            bottom_grid[i].walls['left'] = False
            bottom_grid[i].walls['right'] = False
        for i in range(len(self.initial_grid)):
            for j in range(len(self.initial_grid[i])):
                self.initial_grid[i][j].row += 1
                self.initial_grid[i][j].col += 1
        new_grid = []
        new_grid.append(top_grid)
        for i in range(len(self.initial_grid)):
            new_grid.append([left_grid[i]] + self.initial_grid[i] + [right_grid[i]])
        new_grid.append([left_grid[-1]] + bottom_grid + [right_grid[-1]])
        self.initial_grid = new_grid
        self.num_cols += 2
        self.num_rows+=2
        self.grid_size = self.num_cols * self.num_rows
    
    def make_exits(self):
        for exit in self.exits:
            self.initial_grid[exit[0][0]][exit[0][1]].remove_walls(exit[1][0], exit[1][1])
            self.initial_grid[exit[1][0]][exit[1][1]].remove_walls(exit[0][0], exit[0][1])

import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.DEBUG)


class Visualizer(object):
    """Class that handles all aspects of visualization.


    Attributes:
        maze: The maze that will be visualized
        cell_size (int): How large the cells will be in the plots
        height (int): The height of the maze
        width (int): The width of the maze
        ax: The axes for the plot
        lines:
        squares:
        media_filename (string): The name of the animations and images

    """
    def __init__(self, maze, cell_size, media_filename):
        self.maze = maze
        self.cell_size = cell_size
        self.height = maze.num_rows * cell_size
        self.width = maze.num_cols * cell_size
        self.ax = None
        self.lines = dict()
        self.squares = dict()
        self.media_filename = media_filename

    def set_media_filename(self, filename):
        """Sets the filename of the media
            Args:
                filename (string): The name of the media
        """
        self.media_filename = filename

    def show_maze(self):
        """Displays a plot of the maze without the solution path"""

        # Create the plot figure and style the axes
        fig = self.configure_plot()

        # Plot the walls on the figure
        self.plot_walls()

        # Display the plot to the user
        plt.show()

        # Handle any potential saving
        if self.media_filename:
            fig.savefig("{}{}.png".format(self.media_filename, "_generation"), frameon=None)

    def plot_walls(self):
        """ Plots the walls of a maze. This is used when generating the maze image"""
        for i in range(self.maze.num_rows):
            for j in range(self.maze.num_cols):
                # if self.maze.initial_grid[i][j].is_entry_exit == "entry":
                #     self.ax.text(j*self.cell_size, i*self.cell_size, "START", fontsize=7, weight="bold")
                # elif self.maze.initial_grid[i][j].is_entry_exit == "exit":
                #     self.ax.text(j*self.cell_size, i*self.cell_size, "END", fontsize=7, weight="bold")
                if self.maze.initial_grid[i][j].walls["top"]:
                    self.ax.plot([j*self.cell_size, (j+1)*self.cell_size],
                                 [i*self.cell_size, i*self.cell_size], color="k")
                if self.maze.initial_grid[i][j].walls["right"]:
                    self.ax.plot([(j+1)*self.cell_size, (j+1)*self.cell_size],
                                 [i*self.cell_size, (i+1)*self.cell_size], color="k")
                if self.maze.initial_grid[i][j].walls["bottom"]:
                    self.ax.plot([(j+1)*self.cell_size, j*self.cell_size],
                                 [(i+1)*self.cell_size, (i+1)*self.cell_size], color="k")
                if self.maze.initial_grid[i][j].walls["left"]:
                    self.ax.plot([j*self.cell_size, j*self.cell_size],
                                 [(i+1)*self.cell_size, i*self.cell_size], color="k")

    def configure_plot(self):
        """Sets the initial properties of the maze plot. Also creates the plot and axes"""

        # Create the plot figure
        fig = plt.figure(figsize = (7, 7*self.maze.num_rows/self.maze.num_cols))

        # Create the axes
        self.ax = plt.axes()

        # Set an equal aspect ratio
        self.ax.set_aspect("equal")

        # Remove the axes from the figure
        self.ax.axes.get_xaxis().set_visible(True)
        self.ax.axes.get_yaxis().set_visible(True)

        title_box = self.ax.text(0, self.maze.num_rows + self.cell_size + 0.1,
                            r"{}$\times${}".format(self.maze.num_rows - 2, self.maze.num_cols - 2),
                            bbox={"facecolor": "gray", "alpha": 0.5, "pad": 4}, fontname="serif", fontsize=15)

        return fig
    