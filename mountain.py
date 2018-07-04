import gym.spaces
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense 
from keras.optimizers import Adam
from collections import deque
import random
import os

def classic_model(state_shape, n_actions, weight_backup):
    model = Sequential()
    model.add(Dense(24, input_dim = state_shape, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(n_actions, activation='linear'))
    #optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    optimizer=Adam(lr=0.00025)
    model.compile(optimizer, loss='mse')

    if os.path.isfile(weight_backup):
        model.load_weights(weight_backup)
    return model

class Bricks:
    def __init__(self):
        self.weight_file = "weight2.wt"
        self.sample_batch_size = 32
        self.episodes = 10000
        #self.episodes = 5
        self.env = gym.make('MountainCar-v0')
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.model = classic_model(self.state_size, self.action_size, self.weight_file)
        self.memory = deque(maxlen=1000000)
        self.gamma = 0.95
        self.exploration_rate = 1.0
        self.exploration_min = 0.01
        self.exploration_decay = 0.995
        self.tot_index = 0

    def save_model(self):
        self.model.save(self.weight_file)

    def get_epsilon_for_iteration(self):
        epsilon = 1 + (-1 / 5000) * self.tot_index
        if epsilon < 0.1:
            return 0.1
        return epsilon

    def act(self, state):
        env = self.env
        if np.random.rand() < self.exploration_rate:
            return env.action_space.sample()
        else:
            return np.argmax(self.model.predict(state)[0])

    def replay(self):
        if len(self.memory) < self.sample_batch_size:
            return
        sample_batch = random.sample(self.memory, self.sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.max(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

    def run(self):
        for index_episode in range (0, self.episodes):
            #print("Iteration start: ", env.action_space.n)
            state = self.env.reset()
            done = False
            index = 0
            tot_reward = 0
            #print ("State, env shape: ", state.shape, self.env.observation_space.shape)
            while not done:
                #self.env.render()
                state = np.reshape(state, [1, self.state_size])
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])

                self.memory.append((state, action, reward, next_state, done))
                index += 1
                tot_reward += reward
                
            self.tot_index += 1
            print("Episode {}# Score: {}".format(index_episode, index + 1))
            #print("Episode {}# Score: {}, Reward: {}, Total frames seen: {}, Epsilon: {}, memory_length: {}".format(index_episode, index + 1, tot_reward, self.tot_index, self.get_epsilon_for_iteration(),len(self.memory)))
            self.replay()
        self.save_model()

if __name__ == "__main__":
    bricks = Bricks()
    bricks.run()

