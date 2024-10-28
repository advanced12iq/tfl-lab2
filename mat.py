from maze import Maze
import sys

class DFA:
    def __init__(self, states : set, alphabet : list, transitions : set, start_state : tuple, accept_states : set):
        self.states = states  # множество состояний
        self.alphabet = alphabet  # алфавит
        self.transitions = transitions  # таблица переходов
        self.start_state = start_state  # начальное состояние
        self.accept_states = accept_states  # конечные состояния
        self.current_state = start_state  # текущее состояние

    def reset(self):
        self.current_state = self.start_state  # сброс к начальному состоянию

    def process(self, input_string : str) -> bool:
        for symbol in input_string:
            if symbol in self.alphabet:
                self.current_state = self.transitions.get((self.current_state, symbol), None)
                if self.current_state is None:
                    return False  # недопустимый переход
            else:
                return False  # символ не в алфавите
        return self.current_state in self.accept_states  # проверка на конечное состояние
    


def maze_to_dka(maze : Maze) -> DFA:
    rows = maze.num_rows
    cols = maze.num_cols

    transitions = {}
    start_state = (1, 1)
    accept_states = set([(0, i) for i in range(cols)] + [(rows-1, i) for i in range(cols)] + [(i, 0) for i in range(rows)] + [(i, cols-1) for i in range(cols)] + [(-1, -1)])
    states = set([(i, j) for i in range(rows) for j in range(cols)] + [(-1, -1)])
    alphabet = ['S', 'N', 'W', 'E']
    
    alp_to_wall = {'S' : 'bottom', 'N':'top', 'W':'left', 'E':'right'}
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for i in range(rows):
        for j in range(cols):
            for (a, b), alp in zip(dirs, alphabet):
                x, y= i +a, j + b
                if x in range(rows) and y in range(cols):
                    if not maze.initial_grid[i][j].walls[alp_to_wall[alp]]:
                        transitions[((i, j), alp)] = (x, y)
                    else:
                        transitions[((i, j), alp)] = (i, j)
                else:
                    transitions[((i, j), alp)] = (-1, -1)
    
    return DFA(states, alphabet, transitions, start_state, accept_states)


def get_table():
    lines = sys.stdin.read()
    lines = list(map(lambda s: s.strip().split()))

    suff = lines[0]
    pref = list(map(lambda s: s[0], lines[1:]))

    values = list(map(lambda s: list(map(bool, s[1:])), lines[1:]))
    words = {}

    for i in range(len(pref)):
        for j in range(len(suff)):
            words[pref[i] + suff[j]] = values[i][j]
    
    return words

def table_to_dka(table : dict, maze : Maze) -> DFA:
    rows = maze.num_rows
    cols = maze.num_cols

    alphabet = ['S', 'N', 'W', 'E']
    alp_to_wall = {'S' : 'bottom', 'N':'top', 'W':'left', 'E':'right'}
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    alp_to_dir = {alp : dir for alp, dir in zip(list(alphabet, dirs))}

    transitions = {}
    start_state = (1, 1)
    accept_states = set()
    states = set([start_state])

    hidden_state = None
    

    for word, value in table.items():
        current_state = start_state
        for letter in word:
            # Обрабатываем выход за границы лабиринта
            if current_state == (-1, -1): 
                next_state = (hidden_state[0] + alp_to_dir[letter][0], hidden_state[1] + alp_to_dir[letter][1])
                if hidden_state in range(rows) and hidden_state in range(cols):
                    current_state = next_state
                    hidden_state = None
                else:
                    hidden_state = next_state
                continue
            if maze.initial_grid[current_state[0]][current_state[1]].walls[alp_to_wall[letter]]:
                transitions[(current_state, letter)] = current_state
            else:
                next_state = (current_state[0] + alp_to_dir[letter][0], current_state[1] + alp_to_dir[letter][1])
                # Выход за лабиринт
                if next_state[0] not in range(rows) or next_state[1] not in range(cols):
                    transitions[(current_state, letter)] = (-1, -1)
                    hidden_state = next_state
                    states.add((-1, -1))
                else:
                    states.add(next_state)
                    transitions[(current_state, letter)] = next_state
                current_state = next_state
        if value:
            accept_states.add(current_state)
    
    return DFA(states, alphabet, transitions, start_state, accept_states)

def isin(word : str, maze_dka : DFA) -> bool:
    current_state = maze_dka.start_state
    for letter in word:
        current_state = maze_dka.transitions[(current_state, letter)]
    return current_state in maze_dka.accept_states


from collections import deque
def bfs(maze_dka : DFA, state : tuple, final : bool)->str:
    deq = deque()
    visited = set()
    deq.append((state, ""))
    while deq:
        for _ in range(len(deq)):
            current_state, way = deq.popleft()
            if (final and current_state in maze_dka.accept_states) or (not final and current_state == maze_dka.start_state):
                return way
            visited.add(current_state)
            for letter in maze_dka.alphabet:
                next_state = maze_dka.transitions[(current_state, letter)]
                if next_state in visited:
                    continue
                deq.append((next_state, way + letter))
    return ""


def counter_word_state(maze_dka : DFA, state : tuple) -> str:
    
    # Если отсутствует особое состояние, возвращаем путь до точки + уход в сторону

    if state == (-1, -1):
        return bfs(maze_dka, (0, 0), False)[::-1] + "N"
    
    # Если отсутствует состояние возвращаем маршрут до выхода через это состояние
    # Первый бфс вернет путь от старта до состояния, второй дфс вернет путь от состояния до финального

    return bfs(maze_dka, state, False)[::-1] + bfs(maze_dka, state, True)


def counter_word_accept_state(maze_dka : DFA, accept_state : tuple) -> str:
    
    # Если отсутствует особое состояние, возвращаем путь до точки + уход в сторону

    if accept_state == (-1, -1):
        return bfs(maze_dka, (0, 0), False)[::-1] + "N"
    
    # Если отсутствует финальное состояние просто проходим от финального состояния до старта

    return bfs(maze_dka, accept_state, False)[::-1]


def counter_word_transition(maze_dka : DFA, transition : tuple) -> str:

    # Если отсутствует переход ищем маршрут до выхода через переход

    return bfs(maze_dka, transition[0][0], False)[::-1] + transition[0][1] + bfs(maze_dka, transition[1], True)


def equal(maze_dka : DFA, table_dka : DFA) -> str:

    for state in maze_dka.states:
        if state not in table_dka.states:
            return counter_word_state(maze_dka, state)
        
    for state in maze_dka.accept_states:
        if state not in table_dka.accept_states:
            return counter_word_state(maze_dka, state)
        
    for transition in maze_dka.transitions:
        if transition not in table_dka.transitions:
            return counter_word_state(maze_dka, transition)
        
    return "TRUE"