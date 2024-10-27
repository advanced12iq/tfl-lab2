from maze import Maze
from mat import maze_to_dka, isin, equal, get_table, table_to_dka

def main():
    maze = Maze(30, 30)
    maze_dka = maze_to_dka(maze)
    while True:
        type = input()
        if type == 'isin':
            word = input()
            print(isin(word, maze_dka))
        else:
            table = get_table()
            print(equal(maze_dka, table_to_dka(table)))

if __name__ == '__main__':
    main()