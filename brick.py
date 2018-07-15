import gym.spaces
import numpy as np
import keras
from collections import deque
import random
import os

def transform_reward(reward):
    return np.sign(reward)

def grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)

def downsize(img):
    return img[::2,::2]

def preprocess_image(img):
    return grayscale(downsize(img))

def atari_model(n_actions, weight_backup):
    ATARI_SHAPE = (105, 80, 4)

    frames_input = keras.layers.Input(ATARI_SHAPE, name='frames')
    actions_input = keras.layers.Input((n_actions,), name='mask')

    normalized = keras.layers.Lambda(lambda x: x / 255.0)(frames_input)
    conv_1 = keras.layers.convolutional.Conv2D(16, (8, 8), activation="relu", strides=(4, 4))(normalized)
    conv_2 = keras.layers.convolutional.Conv2D(32, (4, 4), activation="relu", strides=(2, 2))(conv_1)
    conv_flattened = keras.layers.core.Flatten()(conv_2)

    hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)
    output = keras.layers.Dense(n_actions)(hidden)
    filtered_output = keras.layers.merge([output, actions_input], mode='mul')

    #print ("model state shape: ", frames_input.shape)
    #print ("model action shape: ", actions_input.shape)

    model = keras.models.Model(input=[frames_input, actions_input], output= filtered_output)
    optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    model.compile(optimizer, loss='mse')

    if os.path.isfile(weight_backup):
        model.load_weights(weight_backup)
    return model

#memory.append((state, action, reward,downsized_frame, is_done))
def sample_batch(memory, batch_size):
    indices = np.random.permutation(len(memory))[:batch_size]
    cols = [[], [], [], [], []]
    for idx in indices:
        new_memory = memory[idx]
        for col, value in zip(cols, new_memory):
            col.append(value)
    cols = [np.array(col) for col in cols]
    return (cols[0], cols[1], cols[2], cols[3], cols[4])

class Bricks:
    def __init__(self):
        self.weight_file = "weight.wt"
        self.sample_batch_size = 32
        self.episodes = 1000000
        #self.episodes = 5
        self.env = gym.make('BreakoutDeterministic-v4')
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.model = atari_model(self.action_size, self.weight_file)
        self.memory = deque(maxlen=100000)
        self.gamma = 0.95
        self.exploration_rate = 1.0
        self.exploration_min = 0.01
        self.exploration_decay = 0.995
        self.tot_index = 0

    def save_model(self):
        self.model.save(self.weight_file)

    def get_epsilon_for_iteration(self):
        #return (1000000 - self.tot_index) / 1000000
        epsilon = 1.0 + (-1 / 100000) * self.tot_index
        if epsilon < 0.1:
            return 0.1
        return epsilon

    def act(self, state):
        epsilon = self.get_epsilon_for_iteration()
        #epsilon = 0.01
        env = self.env
        #print ("State shape: ", state.shape)
        if np.random.rand() < epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(self.predict(state))

    def predict(self, state):
        state = np.expand_dims(state, axis = 0)
        action_mask = np.ones(self.env.action_space.n)
        action_mask = np.expand_dims(action_mask, axis=0)

        return self.model.predict([state, action_mask])

    def replay(self):
        if len(self.memory) < self.sample_batch_size:
            return
        samples = random.sample(self.memory, self.sample_batch_size)
        #samples = sample_batch(self.memory, self.sample_batch_size)
        #print ("samples: " , samples)
        #i = 0
        for state, action, reward, next_state, done in samples:
            target = reward
            if not done:
                target = reward + self.gamma * np.max(self.predict(next_state)[0])
            target_f = self.predict(state)
            target_f[0][action] = target

            #current_image = state
            #current_image = np.expand_dims(state, axis=0)
            state = np.expand_dims(state, axis = 0)
            action_mask = np.ones(self.env.action_space.n)
            action_mask = np.expand_dims(action_mask, axis=0)

            self.model.fit([state, action_mask], target_f, epochs=1, verbose=0)

    def init_state(self, image):
        return np.stack((image, image, image, image), axis=-1)
    
    def make_state(self, state, image):
        return np.stack((state[:,:,1], state[:,:,2], state[:,:,3], image), axis=-1)

    def run(self):
        for index_episode in range (0, self.episodes):
            #print("Iteration start: ", env.apreprocessed_imagepreprocessed_imagection_space.n)
            image = self.env.reset()
            preprocessed_image = preprocess_image(image)
            state = self.init_state(preprocessed_image)
            done = False
            index = 0
            tot_reward = 0
            while not done:
                preprocessed_image = preprocess_image(image)
                action = self.act(state)
                next_image, reward, done, _ = self.env.step(action)

                preprocessed_next_image = preprocess_image(next_image)
                next_state = self.make_state(state, preprocessed_next_image)

                reward = transform_reward(reward)
                #self.env.render()

                self.memory.append((state, action, reward, next_state, done))
                state = next_state
                index += 1
                tot_reward += reward
                
            self.tot_index += 1
            print("Episode {}# Score: {}, Reward: {}, Total frames seen: {}, Epsilon: {}, memory_length: {}".format(index_episode, index + 1, tot_reward, self.tot_index, self.get_epsilon_for_iteration(),len(self.memory)))
            if index_episode > 50000:
                self.replay()
            if index_episode % 30000 == 0:
                self.save_model()
        #self.save_model()

if __name__ == "__main__":
    bricks = Bricks()
    bricks.run()

