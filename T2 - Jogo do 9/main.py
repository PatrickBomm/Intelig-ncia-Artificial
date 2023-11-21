from collections import deque
import pandas as pd
import heapq
import time

def solveBreadthFirst(initialMatrix, finalMatrix, x, y, count):
    queue = deque([(matrix_to_str(initialMatrix), x, y, [])])
    visited = set()
    while queue:
        matrix_str, x, y, path = queue.popleft()
        matrix = [list(map(int, row)) for row in zip(*[iter(matrix_str)]*3)]
        if matrix == finalMatrix:
            return path, count

        for move_x, move_y in get_possible_moves(x, y):
            if move_space(matrix, x, y, move_x, move_y):
                new_state = matrix_to_str(matrix)
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, move_x, move_y, path + [(move_x, move_y)]))
                    count += 1
                move_space(matrix, move_x, move_y, x, y)


def solveDepthFirst(initialMatrix, finalMatrix, x, y, count):
    stack = [(matrix_to_str(initialMatrix), x, y, [])]
    visited = set()
    while stack:
        matrix_str, x, y, path = stack.pop()
        matrix = [list(map(int, row)) for row in zip(*[iter(matrix_str)]*3)]

        if matrix == finalMatrix:
            return path, count

        for move_x, move_y in get_possible_moves(x, y):
            if move_space(matrix, x, y, move_x, move_y):
                new_state = matrix_to_str(matrix)
                if new_state not in visited:
                    visited.add(new_state)
                    stack.append((new_state, move_x, move_y, path + [(move_x, move_y)]))
                    count += 1
                move_space(matrix, move_x, move_y, x, y)


def manhattan_distance(matrix, finalMatrix):
    distance = 0
    size = len(matrix)
    for i in range(size):
        for j in range(size):
            if matrix[i][j] != 0:
                for x in range(size):
                    for y in range(size):
                        if finalMatrix[x][y] == matrix[i][j]:
                            distance += abs(x - i) + abs(y - j)
                            break
    return distance



def matrix_to_tuple(matrix):
    return tuple(map(tuple, matrix))

def matrix_to_str(matrix):
    return ''.join(map(str, sum(matrix, [])))

def copy_matrix(matrix):
    """ Cria uma cópia profunda de uma matriz. """
    return [row[:] for row in matrix]

def is_valid_move(x, y, max_x, max_y):
    """ Verifica se o movimento é válido dentro das dimensões do tabuleiro. """
    return 0 <= x < max_x and 0 <= y < max_y

def move_space(matrix, x, y, new_x, new_y):
    """ Move o espaço vazio para uma nova posição, se válida. """
    if is_valid_move(new_x, new_y, len(matrix), len(matrix[0])):
        matrix[x][y], matrix[new_x][new_y] = matrix[new_x][new_y], matrix[x][y]
        return True
    return False

def get_possible_moves(x, y):
    """ Retorna uma lista de movimentos possíveis para o espaço vazio. """
    return [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]


def print_matrix(matrix):
    for row in matrix:
        print(' '.join(str(x) for x in row))
    print()

def print_path(path, initialMatrix):
    matrix = copy_matrix(initialMatrix)
    print("Estado Inicial:")
    print_matrix(matrix)
    for x, y in path:
        move_space(matrix, x, y, x, y)
        print_matrix(matrix)

def find_empty_space(matrix):
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if value == 0:
                return i, j
    return -1, -1

def solveGreedyBestFirst(initialMatrix, finalMatrix, x, y, count):
    def heuristic(matrix):
        return manhattan_distance(matrix, finalMatrix)

    priority_queue = [(heuristic(initialMatrix), 0, initialMatrix, x, y, [])]
    visited = set()

    while priority_queue:
        _, cost, matrix, x, y, path = heapq.heappop(priority_queue)

        if matrix == finalMatrix:
            return path, count

        for move_x, move_y in get_possible_moves(x, y):
            new_matrix = copy_matrix(matrix)
            if move_space(new_matrix, x, y, move_x, move_y):
                new_state = (tuple(map(tuple, new_matrix)), move_x, move_y)
                if new_state not in visited:
                    visited.add(new_state)
                    heapq.heappush(priority_queue, (heuristic(new_matrix), cost + 1, new_matrix, move_x, move_y, path + [(move_x, move_y)]))
                    count += 1


def solveAStar(initialMatrix, finalMatrix, x, y, count):
    def heuristic(matrix):
        return manhattan_distance(matrix, finalMatrix)

    priority_queue = [(heuristic(initialMatrix), 0, initialMatrix, x, y, [])]
    visited = set()

    while priority_queue:
        h, cost, matrix, x, y, path = heapq.heappop(priority_queue)

        if matrix == finalMatrix:
            return path, count

        for move_x, move_y in get_possible_moves(x, y):
            new_matrix = copy_matrix(matrix)
            if move_space(new_matrix, x, y, move_x, move_y):
                new_state = (tuple(map(tuple, new_matrix)), move_x, move_y)
                if new_state not in visited:
                    visited.add(new_state)
                    total_cost = cost + 1 + heuristic(new_matrix)
                    heapq.heappush(priority_queue, (total_cost, cost + 1, new_matrix, move_x, move_y, path + [(move_x, move_y)]))
                    count += 1


def comparar_algoritmos(resultados):
    # resultados é um dicionário contendo informações sobre cada algoritmo e tabuleiro
    # Exemplo: resultados['BFS'][0] contém o número de nodos criados pelo BFS no Tabuleiro 1

    analise = {
        "Tabuleiro": [],
        "BFS": [],
        "DFS": [],
        "Greedy": [],
        "A*": []
    }

    # Comparar os algoritmos para cada tabuleiro
    for i in range(len(initialMatrices)):
        analise["Tabuleiro"].append(f"Tabuleiro {i + 1}")
        total_nodos = sum(resultados[algo][i] for algo in ["BFS", "DFS", "Greedy", "A*"])

        # Analisar a porcentagem de nodos utilizados por cada algoritmo
        for algo in ["BFS", "DFS", "Greedy", "A*"]:
            porcentagem_nodos = (resultados[algo][i] / total_nodos) * 100 if total_nodos else 0
            analise[algo].append(f"{porcentagem_nodos:.2f}%")

    # Exibir a análise
    df_analise = pd.DataFrame(analise)
    print(df_analise)

initialMatrices = [
    [[1, 2, 3],
     [4, 5, 6],
     [0, 7, 8]],  # Tabuleiro 1
    [[1, 3, 0],
     [4, 2, 5],
     [7, 8, 6]],  # Tabuleiro 2
    [[1, 3, 5],
     [2, 6, 0],
     [4, 7, 8]],  # Tabuleiro 3
    [[1, 8, 3],
     [4, 2, 6],
     [7, 5, 0]],  # Tabuleiro 4
    [[1, 2, 3],
     [7, 0, 6],
     [4, 8, 5]]   # Tabuleiro 5
]
[(2, 1), (1, 1), (0, 1), (0, 2), (1, 2), (1, 1), (2, 1), (2, 2), (1, 2), (0, 2), (0, 1), (1, 1), (2, 1), (2, 2)]
finalMatrix = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

resultados = {
    "Tabuleiro": [],
    "BFS": [],
    "DFS": [],
    "Greedy": [],
    "A*": []
}

# Executando as funções para cada tabuleiro
for i, initialMatrix in enumerate(initialMatrices):
    x, y = find_empty_space(initialMatrix)

    # Inicializa os contadores para cada algoritmo
    count_bfs = count_dfs = count_greedy = count_astar = 0

    # Executa os algoritmos
    _, count_bfs = solveBreadthFirst(initialMatrix, finalMatrix, x, y, count_bfs)
    _, count_dfs = solveDepthFirst(initialMatrix, finalMatrix, x, y, count_dfs)
    _, count_greedy = solveGreedyBestFirst(initialMatrix, finalMatrix, x, y, count_greedy)
    _, count_astar = solveAStar(initialMatrix, finalMatrix, x, y, count_astar)

    # Armazena os resultados
    resultados["Tabuleiro"].append(f"Tabuleiro {i + 1}")
    resultados["BFS"].append(count_bfs)
    resultados["DFS"].append(count_dfs)
    resultados["Greedy"].append(count_greedy)
    resultados["A*"].append(count_astar)

    # Exibe os caminhos encontrados
    print(f"\n\nTabuleiro {i + 1}")
    print("Caminho BFS:", solveBreadthFirst(initialMatrix, finalMatrix, x, y, count_bfs)[0])
    print("Caminho DFS:", solveDepthFirst(initialMatrix, finalMatrix, x, y, count_dfs)[0])
    print("Caminho Greedy:", solveGreedyBestFirst(initialMatrix, finalMatrix, x, y, count_greedy)[0])
    print("Caminho A*:", solveAStar(initialMatrix, finalMatrix, x, y, count_astar)[0])

# Criando um DataFrame do pandas para visualizar os resultados
df = pd.DataFrame(resultados)
print(df)

comparar_algoritmos(resultados)