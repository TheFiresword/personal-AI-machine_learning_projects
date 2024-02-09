"""
Tic Tac Toe Player
"""

import math, copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    if board is initial_state():
        return X
    countX = [board[i][j]  for i in range(3) for j in range(3) if board[i][j] == X]
    countO = [board[i][j]  for i in range(3) for j in range(3) if board[i][j] == O]
    return O if len(countX) > len(countO) else X




def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    return set([(i,j)  for i in range(3) for j in range(3) if board[i][j] == EMPTY])


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """

    if action not in actions(board):
        raise ValueError
    modified_board = copy.deepcopy(board)
    modified_board[action[0]][action[1]] = player(board)
    return modified_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    winable = [i for i in range(3) if board[i][0] == board[i][1] == board[i][2] != EMPTY]
    if winable:
        return board[winable[0]][0]
    winable = [j for j in range(3) if board[0][j] == board[1][j] == board[2][j] != EMPTY]
    if winable:
        return board[0][winable[0]]
    if board[0][0] == board[1][1] == board[2][2]!= EMPTY or board[0][2] == board[1][1] == board[2][0]!= EMPTY:
        return board[1][1]
    return None

test = [[X, EMPTY, O],
            [EMPTY, X, EMPTY],
            [EMPTY, EMPTY, O]]

def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) or not actions(board):
        return True
    return False



def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    _winner = winner(board)
    return 1 if _winner is X else -1 if _winner is O else 0



def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None
    if board == initial_state():
        return 0,0
    scores = {}
    my_actions = actions(board)
    the_player = player(board)
    for one_action in my_actions:
        scores[one_action] = min_max_value(result(board, one_action))
        if the_player is X and scores[one_action] == 1 or (the_player is O and scores[one_action] == -1):
            break
    return max(scores, key=scores.get) if the_player is X else min(scores, key=scores.get)


def min_max_value(board):
    if terminal(board):
        return utility(board)
    else:
        the_player = player(board)
        future_boards = [result(board, one_action) for one_action in actions(board)]
        if the_player is X:
            return max([min_max_value(choice) for choice in future_boards])
        else:
            return min([min_max_value(choice) for choice in future_boards])
