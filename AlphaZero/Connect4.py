import numpy as np


class Connect4Game:

    def __init__(self):
        self.columns = 7
        self.rows = 6
        

    def get_init_board(self):
        b = np.zeros((self.rows,self.columns), dtype=np.double)
        return b

    def get_board_size(self):
        return self.rows,self.columns

    def get_action_size(self):
        return self.columns

    def drop_piece(self,board,row,col,player):
        board[row][col]= player
    
    def is_valid_location(self,board,col):
        return board[self.rows-1][col]==0

    def get_next_open_row(self,board,col):
        for r in range(self.rows):
            if board[r][col]==0:
                return r

    def print_board(self,board):
        print(np.flip(board,0))

    
    def get_next_state(self, board, player, action):
        b = np.copy(board)
        
        if self.is_valid_location(b,action):
            row = self.get_next_open_row(b,action)
            self.drop_piece(b,row,action,player)

        # Return the new game, but
        # change the perspective of the game with negative
        return (b, -player)
    
    
    def has_legal_moves(self, board):
        for index in range(self.columns):
            if self.is_valid_location(board,index):
                return True
            
        return False

    def get_valid_moves(self, board):
        # All moves are invalid by default
        valid_moves = [0] * self.get_action_size()

        for index in range(self.columns):
            if self.is_valid_location(board,index):
                valid_moves[index] = 1

        return valid_moves

    def is_win(self,board, player):

        # Check horizontal locations for win
        for c in range(self.columns-3):
            for r in range(self.rows):
                if board[r][c] == player and board[r][c+1] == player and board[r][c+2] == player and board[r][c+3] == player:
                    return True
    
        # Check vertical locations for win
        for c in range(self.columns):
            for r in range(self.rows-3):
                if board[r][c] == player and board[r+1][c] == player and board[r+2][c] == player and board[r+3][c] == player:
                    return True
    
        # Check positively sloped diaganols
        for c in range(self.columns-3):
            for r in range(self.rows-3):
                if board[r][c] == player and board[r+1][c+1] == player and board[r+2][c+2] == player and board[r+3][c+3] == player:
                    return True
    
        # Check negatively sloped diaganols
        for c in range(self.columns-3):
            for r in range(3, self.rows):
                if board[r][c] == player and board[r-1][c+1] == player and board[r-2][c+2] == player and board[r-3][c+3] == player:
                    return True

        return False
        

    def get_reward_for_player(self, board, player):
        # return None if not ended, 1 if player 1 wins, -1 if player 1 lost
        if self.is_win(board, player):
            return 1
        if self.is_win(board, -player):
            return -1
        if self.has_legal_moves(board):
            return None
        return 0

    def get_perspective_board(self, board, player):
        return player * board

if __name__=="__main__":
    game = Connect4Game()

    board = game.get_init_board()
    game.print_board(board)
    game_over = False
    turn = 0
    
    while not game_over:
        #Ask for player 1 input
        if turn == 0:
            col = int(input("Player 1, Make your Selection(0-6):"))
            #Player 1 will drop a piece on the board
            while not game.is_valid_location(board,col):
                print("Move is invalid, try again!")
                col = int(input("Player 1, Make your Selection(0-6):"))
            if game.is_valid_location(board,col):
                row = game.get_next_open_row(board,col)
                game.drop_piece(board,row,col,1)
                
            if game.is_win(board, 1):
                print("Player 1 wins!")
                game_over = True
            
        #Ask for player 2 input
        else:
            col = int(input("Player 2, Make your Selection(0-6):"))
            #Player 2 will drop a piece on the board
            while not game.is_valid_location(board,col):
                print("Move is invalid, try again!")
                col = int(input("Player 2, Make your Selection(0-6):"))
            if game.is_valid_location(board,col):
                row = game.get_next_open_row(board,col)
                game.drop_piece(board,row,col,-1)
            if game.is_win(board, -1):
                print("Player 2 wins!")
                game_over = True

        game.print_board(board)
       
                
        turn += 1
        turn = turn % 2