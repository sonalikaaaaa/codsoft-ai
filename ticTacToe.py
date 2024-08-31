import math

#constants for the game
player_x = 'X'
player_o = 'O'
EMPTY = ' '

#function to print board
def print_board(board):
    for row in board:
        print(' | '.join(row))
        print('-' * 10)

#function to check if there is a winner
def check_winner(board):
    lines = []
    #rows and columns
    lines.extend(board)
    lines.extend([[board[i][j] for i in range(3)] for j in range(3)])

    #diagonals
    lines.append([board[i][i] for i in range(3)])
    lines.append([board[i][2 - i] for i in range(3)])

    for line in lines:
        if line.count(player_x) == 3:
            return player_x
        if line.count(player_o) == 3:
            return player_o
        
    return None

def is_full(board):
    return all(cell != EMPTY for row in board for cell in row)

#function to evaluate board state
def evaluate(board):
    winner = check_winner(board)
    if winner == player_x:
        return 10
    elif winner == player_o:
        return -10
    return 0

#minimax algo with alpha-beta pruning
def minimax(board, depth, alpha, beta, is_max):
    score = evaluate(board)
    if score == 10:
        return score - depth
    if score == -10:
        return score + depth
    if is_full(board):
        return 0
    
    if is_max: #for player_x
        best = -math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == EMPTY:
                    board[i][j] = player_x
                    best = max(best, minimax(board, depth + 1, alpha, beta, not is_max))
                    board[i][j] = EMPTY
                    alpha = max(alpha, best)
                    if beta <= alpha:
                        break
        return best
    else: #for player_o
        best = math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == EMPTY:
                    board[i][j] = player_o
                    best = min(best, minimax(board, depth + 1, alpha, beta, not is_max))
                    board[i][j] = EMPTY
                    beta = min(beta, best)
                    if beta <= alpha:
                        break
        return best
    
def find_best_move(board):
    best_val = -math.inf
    best_move = (-1, -1)

    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                board[i][j] = player_x
                move_val = minimax(board, 0, -math.inf, math.inf, False)
                board[i][j] = EMPTY
                if move_val > best_val:
                    best_move = (i, j)
                    best_val = move_val

    return best_move

#Main game loop
def play_game():
    board = [[EMPTY] * 3 for _ in range(3)]
    print("Tic-Tac-Toe Game")
    print_board(board)

    while True:
        if is_full(board):
            print("The game is a draw")
            break

        #Human Move
        x, y = map(int, input("Enter your move (row and column): ").split())
        if board[x][y] == EMPTY:
            board[x][y] = player_o
        else:
            print("Invalid move. Try again")
            continue

        if check_winner(board):
            print_board(board)
            print("You win!")
            break

        if is_full(board):
            print_board(board)
            print("This game is a draw")
            break

        #AI Move
        print("AI making a move...")
        move = find_best_move(board)
        if move != (-1, -1):
            board[move[0]][move[1]] = player_x
        print_board(board)

        if check_winner(board):
            print("AI wins!")
            break

if __name__ == "__main__":
    play_game()