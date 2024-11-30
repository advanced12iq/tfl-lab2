from maze import Maze, Visualizer
from mat import maze_to_dka, isin, equal, get_table, table_to_dka, new_table_to_dka
import random
from dotenv import load_dotenv
import os

load_dotenv()

RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))  # Default to 42 if not set
HEIGHT = int(os.getenv("HEIGHT", 10))
WIDTH = int(os.getenv("WIDTH", 10))

random.seed(RANDOM_SEED)

def main():
    maze = Maze(HEIGHT, WIDTH)
    maze_dka = maze_to_dka(maze)
    # for key, value in maze_dka.transitions.items():
    #     print(key, value)
    # vis = Visualizer(maze, 1, "")
    # vis.show_maze()
    while True:
        type = input()
        if type == 'isin':
            word = input()
            print(isin(word, maze_dka, maze))
        elif type == 'table':
            table = get_table()
            table_dka = new_table_to_dka(table, maze)
            print(equal(maze_dka, table_dka))
        elif type == 'end':
            break

if __name__ == '__main__':
    main()
