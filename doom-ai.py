import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import TruncatedNormal
import tensorflow.keras as K
import numpy as np

from collections import deque
import itertools as it
import cv2
from vizdoom import *   
import random
import time
import datetime

def prepare_gpu():
    print("Initializing GPUs...")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        except RuntimeError as e:
            print(e)
    print("Finsihed starting GPUs")


def clean_gpu():
    from numba import cuda
    cuda.select_device(0)
    cuda.close()

############################################################ AUX Functions ################################################################################

def preprocess(img):
    #img = np.reshape(img,(img.shape[1],img.shape[2],img.shape[0]))
    img = cv2.resize(img, (resolution[1],resolution[0]))
    img = img.astype(np.float32)
    img = img / 255
    return img

def create_model(n_actions):
    model = Sequential()
    model.add(Conv2D(8,6,input_shape=(4, resolution[0], resolution[1]), activation='relu', padding='same', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None)))
    model.add(Conv2D(8,6, activation='relu', padding='same', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None)))
    model.add(Dense(n_actions, kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None)))

    opt = Adam(lr=0.001)
    model.compile(opt,'mean_squared_error',['accuracy'])

    return model    


############################################################ DQN Agent ################################################################################


class DQNAgent:
    def __init__(self, n_actions, max_replay_memory, min_replay_memory, minibatch_size, actions, use_latest=False):
        print("Initializing agent...")
        self.model = create_model(n_actions)
        self.replay_memory = deque(maxlen=max_replay_memory)
        self.min_replay_memory = min_replay_memory
        self.minibatch_size = minibatch_size
        self.actions = actions
        if use_latest:
            self.load_model()
        
        print("Finished the agent!")


    def save_model(self):
        self.model.save("model")

    def load_model(self):
        self.model = K.models.load_model("model")

    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, states):
        states_exp = np.expand_dims(states, axis=0)
        prediction = self.model.predict(states_exp)
        return prediction[0]

    # Trains network every step during episode
    def train(self):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < self.min_replay_memory:
            return None
        
        minibatch = random.sample(self.replay_memory, self.minibatch_size)

        s1_batch = np.array([d[0] for d in minibatch])
        a_batch = [d[1] for d in minibatch]
        r_batch = [d[2] for d in minibatch]
        s2_batch = np.array([d[3] for d in minibatch])

        Y = []
        s2_qs = self.model.predict(s2_batch)

        for i in range(0,self.minibatch_size):
            value = 0
            # Check if terminal
            if minibatch[i][4]:
                value = r_batch[i]
            else:
                value = r_batch[i] + GAMMA * np.max(s2_qs[i])
            tmp = np.zeros(len(self.actions))
            tmp[minibatch[i][1]] = value
            Y.append(tmp)
        
        history = self.model.fit(s1_batch, np.array(Y), batch_size=self.minibatch_size, epochs=1, verbose=0, shuffle=False)
        return history
    
    
############################################################ Environment ################################################################################

    
class Environment:
    def __init__(self, gamma, epsilon, epsilon_disc, max_replay_memory, min_replay_memory, minibatch_size, use_trained=False):
        self.game, self.actions = self.configure_game_training()
        self.agent   = DQNAgent(len(self.actions), max_replay_memory, min_replay_memory, minibatch_size, self.actions, use_latest=use_trained)
        self.gamma = gamma
        self.epsilon_base = epsilon
        self.epsilon_disc = epsilon_disc
        self.epsilon_min = 0.1
        

    def configure_game_training(self):
        print("Initializing game environment...")
        game = DoomGame()
        game.load_config("/home/msi-gtfo/repos/ViZDoom/scenarios/basic.cfg")
        game.set_window_visible(False)
        #game.set_render_hud(False)
        game.set_screen_format(vizdoom.ScreenFormat.GRAY8)

        nothing     = [0, 0, 0]
        left        = [1, 0, 0]
        right       = [0, 1, 0]
        shoot       = [0, 0, 1]
        left_shoot  = [1, 0, 1]
        right_shoot = [0, 1, 1]
        possible_actions = [nothing, left, right, shoot, left_shoot, right_shoot]

        print("Finished game environment!")
        return game, possible_actions

    def train_agent(self, epochs, max_steps):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = 'logs/dqn/' + current_time
        summary_writer = tf.summary.create_file_writer(log_dir)
        
        
        scores = []
        print("---------------Starting training Doom Ai--------------")
        self.game.init()
        epsilon = self.epsilon_base
        
        for epoch in range(epochs):
            losses = []
            print("-> Episode ",epoch)
            self.game.new_episode()
            
            # First state to predict on as a starting point
            s1 = preprocess(self.game.get_state().screen_buffer)
            s_t_deque = deque(maxlen=4)
            s_t_deque.append(s1)
            s_t_deque.append(s1)
            s_t_deque.append(s1)
            s_t_deque.append(s1)

            s_t = np.stack(s_t_deque, axis=0)
            step = 0
            while (not self.game.is_episode_finished()) or step < max_steps:
                q_t = self.agent.get_qs(s_t) 

                # Decide if greedy or not
                if random.random() <= epsilon:
                    a = random.randint(0, len(self.actions) - 1)
                else:
                    a = np.argmax(q_t)

                # Execute action    
                reward = self.game.make_action(self.actions[a],12) #frame repeat ?
                isterminal = self.game.is_episode_finished()
                s2 = preprocess(self.game.get_state().screen_buffer) if not isterminal else np.zeros((resolution))
                s_t_deque.append(s2)
                s_t2 = np.stack(s_t_deque, axis=0)

                self.agent.update_replay_memory([s_t, a, reward, s_t2, isterminal])

                # Update current input of states
                s_t = s_t2
                history = self.agent.train()
                if history is not None:
                    losses.append(history.history['loss'][0])
                else:
                    losses.append(0)
                step += 1
                epsilon = max(self.epsilon_min, epsilon * self.epsilon_disc)

            if self.game.is_episode_finished():
                final_reward = self.game.get_total_reward()
                print("Final Reward: ", final_reward)
                with summary_writer.as_default():
                    tf.summary.scalar('episode reward', final_reward, step=epoch)
                    tf.summary.scalar('average loss', np.array(losses).mean(), step=epoch)
                    tf.summary.scalar('episode epsilon', epsilon, step=epoch)
                scores.append(final_reward)
                losses = []
            
            if (epoch % 5) == 0:
                self.agent.save_model()
            
        train_scores = np.array(scores)
        return train_scores
    
    def play_agent(self, rounds=5):
        # Reinitialize the game with window visible
        self.game, self.actions = self.configure_game_training()
        self.game.set_window_visible(True)
        self.game.set_mode(vizdoom.Mode.ASYNC_PLAYER)
        self.game.init()

        for r in range(rounds):
            self.game.new_episode()

            s1 = preprocess(self.game.get_state().screen_buffer)
            s_t_deque = deque(maxlen=4)
            s_t_deque.append(s1)
            s_t_deque.append(s1)
            s_t_deque.append(s1)
            s_t_deque.append(s1)

            s_t = np.stack(s_t_deque, axis=2)
            while not self.game.is_episode_finished():
                q_t = this.agent.get_qs(s_t) 
                a = np.argmax(q_t)

                # Instead of make_action(a, frame_repeat) in order to make the animation smooth
                self.game.set_action(this.actions[a])
                for _ in range(12):
                    self.game.advance_action()

                s2 = []
                if not self.game.is_episode_finished:
                    s2 = preprocess(self.game.get_state().screen_buffer)
                else:
                    break
                s_t_deque.append(s2)
                s_t2 = np.stack(s_t_deque, axis=2)

            # Sleep between episodes
            time.sleep(1.0)
            score = self.game.get_total_reward()
            print("Round ", r, ": ", score)
    
    
    
################################################################################################################################################    
################################################################################################################################################    
################################################################################################################################################    

EPOCHS = 10
MAX_STEPS = 100

MIN_REPLAY_MEMORY = 1000
MAX_REPLAY_MEMORY = 50000
MINI_BATCH_SIZE   = 64

GAMMA = 0.99
EPSILON = 0.9
EPSILON_DISCOUNT =0.9999

resolution = (84,84)
    
prepare_gpu()
env = Environment(GAMMA, EPSILON, EPSILON_DISCOUNT, MAX_REPLAY_MEMORY, MIN_REPLAY_MEMORY, MINI_BATCH_SIZE, use_trained=True)

# Train agent
train_scores = env.train_agent(EPOCHS, MAX_STEPS)   
print("Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()),
        "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())  
    
clean_gpu()
