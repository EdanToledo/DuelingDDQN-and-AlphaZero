from Connect4 import Connect4Game
from agent import AlphaZero

            
if __name__ == "__main__":

    EPISODES_OF_SELF_PLAY=75
    EPOCHS=70
    BATCH_SIZE=64
    ITERATIONS=200
    HIDDEN_SIZE=512
    NUM_SIMULATIONS=300
    LEARNING_RATE=5e-4
    SAVE_PATH="./AlphaZero.pt"
    ARENA_GAMES= 40

    game = Connect4Game()

    agent = AlphaZero(game,(game.rows,game.columns),HIDDEN_SIZE,game.columns,NUM_SIMULATIONS,LEARNING_RATE)

    agent.load_weights(SAVE_PATH)

    print("Training...")

    agent.train(EPISODES_OF_SELF_PLAY,EPOCHS,BATCH_SIZE,SAVE_PATH,ITERATIONS,arena_games=ARENA_GAMES)
