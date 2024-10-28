from maze import Maze, Visualizer
from mat import maze_to_dka, isin, equal, get_table, table_to_dka
import random
from dotenv import load_dotenv
import os

load_dotenv()

RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))  # Default to 42 if not set
HEIGHT = int(os.getenv("HEIGHT", 10))
WIDTH = int(os.getenv("WIDTH", 10))

random.seed(5125)

def main():
    maze = Maze(1, 1)
    vis = Visualizer(maze, 1, "")
    vis.show_maze()
    maze_dka = maze_to_dka(maze)
    print(maze.num_exits)
    print(maze.possible_exits)
    print(maze.exits)
    for i in range(len(maze.initial_grid)):
        for j in range(len(maze.initial_grid[i])):
            print((i, j), maze.initial_grid[i][j].walls)
    while True:
        type = input()
        if type == 'isin':
            word = input()
            print(isin(word, maze_dka))
        else:
            table = get_table()
            print(equal(maze_dka, table_to_dka(table, maze)))

if __name__ == '__main__':
    main()