from maze import Maze, Visualizer
from mat import maze_to_dka, isin, equal, get_table, table_to_dka
import random
from dotenv import load_dotenv
import os

load_dotenv()

RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))  # Default to 42 if not set
HEIGHT = int(os.getenv("HEIGHT", 10))
WIDTH = int(os.getenv("WIDTH", 10))

random.seed(2)

def main():
    maze = Maze(1, 1)
    maze_dka = maze_to_dka(maze)
    
    vis = Visualizer(maze, 1, "")
    # vis.show_maze()
    
    while True:
        type = input()
        if type == 'isin':
            word = input()
            print(isin(word, maze_dka, maze))
        else:
            table = get_table()
            print(equal(maze_dka, table_to_dka(table, maze)))

if __name__ == '__main__':
    main()