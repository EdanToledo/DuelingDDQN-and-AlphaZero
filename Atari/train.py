from collections import deque
import gym
import numpy as np
import torch
import argparse
from agent import DQN_Agent
from gym.wrappers import FrameStack, ResizeObservation, GrayScaleObservation, Monitor
import model



def train(agent, env, num_episodes, log_every=20,save_model_name="pong.pt"):
    """
    :param agent: the agent to be trained.
    :param env: the gym environment.
    :param num_episodes: the number of episodes to train.
    :param log_every: The frequency of logging. Default logs every 20 episodes.
    """

    # Running Average Reward Memory
    running_avg_reward = deque(maxlen=10)
    best_running_avg = float('-inf')
    agent.train()
    for episode in range(1, num_episodes + 1):
        # Starting state observation
        obs = torch.tensor(np.array(env.reset()), dtype=torch.float32,device=agent.device).unsqueeze(0).squeeze(-1)
        
        reward_total = 0
        while True:
            
            
            # return chosen action
            action = agent.act(obs)
            
            # Take a step in the environment with the action drawn
            next_obs, reward, done, info = env.step(action.item())
            

            # Just for logging
            reward_total += reward

            # change next state into tensor for update
            next_obs = torch.tensor(np.array(next_obs), dtype=torch.float32,device=agent.device).unsqueeze(0).squeeze(-1)

            # change reward into tensor for update
            reward = torch.tensor(
                reward, dtype=torch.float32,device=agent.device)

            # Store transition in replay memory

            agent.cache(obs, action, reward, next_obs, torch.tensor(done,device=agent.device))

            # Perform update
            agent.update_model()

            # If the number of steps has elapsed then perform update
            if agent.step_no % agent.update_frequency == 0:
                agent.update_target()

            # Set the current state to the next state
            obs = next_obs

            # if done then log and break
            if done:
                
                if episode % log_every == 0 and len(running_avg_reward) > 0:
                    running_avg = sum(running_avg_reward) / len(running_avg_reward)

                    if running_avg > best_running_avg:
                        best_running_avg = running_avg
                        agent.update_best_weights()
                        agent.save_weights(save_model_name)

                    print(
                        "Episode {0:4d} finished with a total reward of {1:3.1f} | Running Average: {2:3.2f}".format(
                            episode, reward_total, running_avg))
                break
        running_avg_reward.append(reward_total)

    env.close()

def eval(agent, env, num_episodes, log_every=20):
    """
    :param agent: the agent to be evaluated.
    :param env: the gym environment.
    :param num_episodes: the number of episodes to train.
    :param log_every: The frequency of logging. Default logs every 20 episodes.
    """

    # Running Average Reward Memory
    running_avg_reward = deque(maxlen=10)
    best_running_avg = float('-inf')
    # agent.eval()
    for episode in range(1, num_episodes + 1):
        # Starting state observation
        obs = torch.tensor(np.array(env.reset()), dtype=torch.float32,device=agent.device).unsqueeze(0).squeeze(-1)
            
        reward_total = 0
        while True:
            
            
            # return chosen action
            action = agent.act(obs)
            
            # Take a step in the environment with the action drawn
            next_obs, reward, done, info = env.step(action.item())
            

            # Just for logging
            reward_total += reward

            # change next state into tensor for update
            next_obs = torch.tensor(np.array(next_obs), dtype=torch.float32,device=agent.device).unsqueeze(0).squeeze(-1)

            # change reward into tensor for update
            reward = torch.tensor(
                reward, dtype=torch.float32,device=agent.device)

            # Set the current state to the next state
            obs = next_obs

            # if done then log and break
            if done:
                
                if episode % log_every == 0 and len(running_avg_reward) > 0:
                    running_avg = sum(running_avg_reward) / len(running_avg_reward)

                    print(
                        "Episode {0:4d} finished with a total reward of {1:3.1f} | Running Average: {2:3.2f}".format(
                            episode, reward_total, running_avg))
                break
        running_avg_reward.append(reward_total)

    env.close()

if __name__ == "__main__":

    # Hyper parameters
    HIDDEN_SIZE = 256
    GAMMA = 0.99
    LEARNING_RATE = 0.00025
    EPISODES_TO_TRAIN = 5000
    EPISODES_TO_EVAL = 1
    LOG_EVERY = 1
    EPSILON_START = 0
    EPSILON_END = 0
    EPSILON_ANNEAL_OVER_STEPS = 10000
    BATCH_SIZE = 32
    UPDATE_FREQUENCY = 1500
    REPLAY_MEMORY_SIZE = 15000
    INPUT_CHANNELS = 4
    USE_DUELING_NETWORK=True
    USE_DDQN=False
    MODEL_NAME= "DQN" if not USE_DDQN and not USE_DUELING_NETWORK else "DuelingDQN" if not USE_DDQN else "DDQN" if not USE_DUELING_NETWORK else "DuelingDDQN"
    LOAD_PREVIOUS_MODEL=False
    PREVIOUS_MODEL_PATH="./DuelingDDQNALEPong-v5.pt"
    TRAIN=True
    EVAL=False
    RENDER=False
    RECORD=False
    
    env = gym.make('ALE/Pong-v5',
                   obs_type='grayscale',  # ram | rgb | grayscale
                   frameskip=5,  # frame skip
                   mode=0,  # game mode, see Machado et al. 2018
                   difficulty=0,  # game difficulty, see Machado et al. 2018
                   repeat_action_probability=0,  # Sticky action probability
                   full_action_space=True,  # Use all actions
                   render_mode="human" if RENDER else None  # None | human | rgb_array
                   )
    
    env = ResizeObservation(env,(84,84))
    if RECORD:
        env = Monitor(env, './video', force=True)
    
    env = FrameStack(env,4)

    nb_actions = env.action_space.n
    


    agent = DQN_Agent(env.observation_space.shape,INPUT_CHANNELS, HIDDEN_SIZE, nb_actions, GAMMA, LEARNING_RATE, EPSILON_START,
                      EPSILON_END, EPSILON_ANNEAL_OVER_STEPS, BATCH_SIZE, UPDATE_FREQUENCY, REPLAY_MEMORY_SIZE,USE_DUELING_NETWORK,USE_DDQN)

    if LOAD_PREVIOUS_MODEL:
        print("Loading Model...")
        agent.load_weights(PREVIOUS_MODEL_PATH)
    
    if TRAIN:
        print("Training...")

        train(agent, env, EPISODES_TO_TRAIN, LOG_EVERY, MODEL_NAME+(env.unwrapped.spec.id).replace("/","")+".pt")

    if EVAL:
        print("Evaluating...")

        eval(agent, env, EPISODES_TO_EVAL, LOG_EVERY)

    
