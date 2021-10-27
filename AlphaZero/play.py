from Connect4 import Connect4Game
from agent import AlphaZero
import pygame
import sys
import math
import agent_old

# Visualisations and game set up taken from https://www.askpython.com/python/examples/connect-four-game


BLUE = (0,0,255)
BLACK = (0,0,0)
RED = (255,0,0)
YELLOW = (255,255,0)
 
def play(game,agent):
    
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
            print("AlphaZero's Turn...")
            col = agent.act(board,-1)
            while not game.is_valid_location(board,col):
                print("Move is invalid, try again!")
                col = agent.act(board,-1)
            if game.is_valid_location(board,col):
                row = game.get_next_open_row(board,col)
                game.drop_piece(board,row,col,-1)
            if game.is_win(board, -1):
                print("AlphaZero Wins!")
                game_over = True

        game.print_board(board)
       
                
        turn += 1
        turn = turn % 2


def play_with_gui(game,agent,temperature=1,computer_match=False,second_agent=None):
    """Play game with GUI"""

    board = game.get_init_board()
    game.print_board(board)
    game_over = False
    turn = 0
    
    #initalize pygame
    pygame.init()
    
    #define our screen size
    SQUARESIZE = 100

    COLUMN_COUNT = game.columns
    ROW_COUNT = game.rows

    #define width and height of board
    width = COLUMN_COUNT * SQUARESIZE
    height = (ROW_COUNT+1) * SQUARESIZE
    
    size = (width, height)
    
    RADIUS = int(SQUARESIZE/2 - 5)
    
    screen = pygame.display.set_mode(size)
    
    #Calling function draw_board again
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, BLUE, (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
     
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):      
            if board[r][c] == 1:
                pygame.draw.circle(screen, RED, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
            elif board[r][c] == 2: 
                pygame.draw.circle(screen, YELLOW, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
    
    pygame.display.update()
    
    myfont = pygame.font.SysFont("monospace", 75)
    
    while not game_over:
    
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if not computer_match:
                if event.type == pygame.MOUSEMOTION:
                    pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))
                    posx = event.pos[0]
                    if turn == 0:
                        pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE/2)), RADIUS)
                    else: 
                        pygame.draw.circle(screen, YELLOW, (posx, int(SQUARESIZE/2)), RADIUS)
                pygame.display.update()
            
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))
                    #print(event.pos)
                    # Ask for Player 1 Input
                    if turn == 0:
                        posx = event.pos[0]
                        col = int(math.floor(posx/SQUARESIZE))
                        
                        if game.is_valid_location(board, col):
                            row = game.get_next_open_row(board, col)
                            game.drop_piece(board, row, col, 1)
        
                            reward = game.get_reward_for_player(board,1)
                            if reward==1:
                                label = myfont.render("Player 1 Wins!", 1, RED)
                                screen.blit(label, (40,10))
                                game_over = True
                            elif reward==0:
                                    label = myfont.render("Draw - nobody wins!", 1, BLUE)
                                    screen.blit(label, (40,10))
                                    game_over = True
                        else:
                            turn=-1
                            
        
                        game.print_board(board)
                        for c in range(COLUMN_COUNT):
                            for r in range(ROW_COUNT):
                                pygame.draw.rect(screen, BLUE, (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
                                pygame.draw.circle(screen, BLACK, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
                        
                        for c in range(COLUMN_COUNT):
                            for r in range(ROW_COUNT):      
                                if board[r][c] == 1:
                                    pygame.draw.circle(screen, RED, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
                                elif board[r][c] == -1: 
                                    pygame.draw.circle(screen, YELLOW, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
                        
                        pygame.display.update()
            
                        turn += 1
                        turn = turn % 2
            
                        if game_over:
                            pygame.time.wait(5000)
                            break
                
            else:
                col = second_agent.act(board,1,best=True,temperature=temperature)

                while not game.is_valid_location(board, col):
                    col = second_agent.act(board,1,best=True,temperature=temperature)

                if game.is_valid_location(board, col):
                    row = game.get_next_open_row(board, col)
                    game.drop_piece(board, row, col, 1)

                    reward = game.get_reward_for_player(board,1)
                    if reward==1:
                        label = myfont.render("AlphaZero-2 Wins!", 1, RED)
                        screen.blit(label, (40,10))
                        game_over = True
                    elif reward==0:
                            label = myfont.render("Draw - nobody wins!", 1, BLUE)
                            screen.blit(label, (40,10))
                            game_over = True

                game.print_board(board)
                for c in range(COLUMN_COUNT):
                    for r in range(ROW_COUNT):
                        pygame.draw.rect(screen, BLUE, (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
                        pygame.draw.circle(screen, BLACK, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
                
                for c in range(COLUMN_COUNT):
                    for r in range(ROW_COUNT):      
                        if board[r][c] == 1:
                            pygame.draw.circle(screen, RED, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
                        elif board[r][c] == -1: 
                            pygame.draw.circle(screen, YELLOW, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
                
                pygame.display.update()

                turn += 1
                turn = turn % 2

                if game_over:
                    pygame.time.wait(5000)
                    break
            if turn ==1:
                
                col = agent.act(board,-1,best=True,temperature=temperature)

                while not game.is_valid_location(board, col):
                    col = agent.act(board,-1,best=True,temperature=temperature)

                if game.is_valid_location(board, col):
                    row = game.get_next_open_row(board, col)
                    game.drop_piece(board, row, col, -1)

                    reward = game.get_reward_for_player(board,-1)
                    if reward==1:
                        label = myfont.render("AlphaZero Wins!", 1, YELLOW)
                        screen.blit(label, (40,10))
                        game_over = True
                    elif reward==0:
                            label = myfont.render("Draw - nobody wins!", 1, BLUE)
                            screen.blit(label, (40,10))
                            game_over = True

                game.print_board(board)
                for c in range(COLUMN_COUNT):
                    for r in range(ROW_COUNT):
                        pygame.draw.rect(screen, BLUE, (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
                        pygame.draw.circle(screen, BLACK, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
                
                for c in range(COLUMN_COUNT):
                    for r in range(ROW_COUNT):      
                        if board[r][c] == 1:
                            pygame.draw.circle(screen, RED, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
                        elif board[r][c] == -1: 
                            pygame.draw.circle(screen, YELLOW, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
                
                pygame.display.update()

                turn += 1
                turn = turn % 2

                if game_over:
                    pygame.time.wait(5000)
                    break

def arena(game,agent1,agent2,temperature=1,number_of_matches=100):
    wins = []
    for i in range(number_of_matches):
        board = game.get_init_board()

        game_over = False
        turn = 0

        while not game_over:
            if turn == 0:
                col = agent1.act(board, 1, False,temperature,0)

                if game.is_valid_location(board, col):
                    row = game.get_next_open_row(board, col)
                    game.drop_piece(board, row, col, 1)

                reward = game.get_reward_for_player(board, 1)

                if reward is not None:
                    wins.append(1 if reward == 1 else 0)
                    game_over = True

            else:

                col = agent2.act(board, -1, False,temperature,0)
                if game.is_valid_location(board, col):
                    row = game.get_next_open_row(board, col)
                    game.drop_piece(board, row, col, -1)

                reward = game.get_reward_for_player(board, -1)

                if reward is not None:
                    wins.append(0 if reward == 1 or reward == 0 else 1)
                    game_over = True

            turn += 1
            turn = turn % 2

    print("Agent 1 Win Percentage:", 100*sum(wins)/number_of_matches)

if __name__ == "__main__":

    HIDDEN_SIZE=512
    NUM_SIMULATIONS=500
    SAVE_PATH="./AlphaZero.pt"

    game = Connect4Game()

    agent = AlphaZero(game,game.rows*game.columns,HIDDEN_SIZE,game.columns,NUM_SIMULATIONS,learning_rate=5e-4)
    agent.load_weights(SAVE_PATH)
    
    agent2 = agent_old.AlphaZero(game,game.rows*game.columns,HIDDEN_SIZE,game.columns,NUM_SIMULATIONS,learning_rate=5e-4)
    agent2.load_weights("./AlphaZero_old.pt")

    play_with_gui(game,agent,0,True,agent2)
    # arena(game,agent,agent2,1)
