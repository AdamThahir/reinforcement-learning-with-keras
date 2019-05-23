from __future__ import division
from PIL import Image

import numpy as np
import argparse
import gym
import keras
import os
import time

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, Convolution2D, Permute
from keras.optimizers import Adam
from keras import backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4
ENV_NAME = 'BreakoutDeterministic-v4'

class BreakoutProcessor (Processor):
    def process_observation (self, observation):
        assert observation.ndim == 3 # (height, width, channel)
        img = Image.fromarray(observation)

        # Resize and convert to grayscale
        img = img.resize(INPUT_SHAPE).convert('L')
        processed_observation = np.array(img)

        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')

    def process_state_batch (self, batch):
        processed_batch = batch.astype('float')/255.
        return processed_batch

    def process_reward (self, reward):
        return np.clip(reward, -1., 1.)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default=None)
    parser.add_argument('--weights', type=str, default=None)
    args = parser.parse_args()

    env = gym.make(ENV_NAME)
    np.random.seed(2019)
    env.seed(2019)

    nHiddenLayers = 5

    if args.weights == None:
        file_path = './weights/dqn_{}_weights_({}).h5f'.format(ENV_NAME, nHiddenLayers)
    else:
        file_path = args.weights

    loadFromExisting = True if os.path.isfile(file_path) else False

    # Related to NN output Layer.
    nb_actions = env.action_space.n


    # Keras Model
    input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE

    # Xavier Uniform Weight Initializer.
    weight_initializer = keras.initializers.glorot_uniform()

    inputLayer = Input(shape=input_shape)
    if K.image_dim_ordering() == 'tf':
        # (width, height, channels)
        hiddenLayer = Permute((2, 3, 1), input_shape=input_shape)(inputLayer)
    elif K.image_dim_ordering() == 'th':
        # (channels, width, height)
        hiddenLayer = Permute((1, 2, 3), input_shape=input_shape)(inputLayer)
    else:
        raise RuntimeError ('Unknown image_dim_ordering')

    # Convolution2D layers
    hiddenLayer = Convolution2D (32, (8, 8), strides=(4, 4), activation='relu')(hiddenLayer)
    hiddenLayer = Convolution2D (64, (4, 4), strides=(2, 2), activation='relu')(hiddenLayer)
    hiddenLayer = Convolution2D (64, (3, 3), strides=(1, 1), activation='relu')(hiddenLayer)

    hiddenLayer = Flatten ()(hiddenLayer)
    hiddenLayer = Dense (512, activation='relu')(hiddenLayer)
    
    outputLayer = Dense (nb_actions, activation='linear')(hiddenLayer)

    model = Model(inputLayer, outputLayer)
    print (model.summary())

    memory      = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    processor   = BreakoutProcessor()

    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                nb_steps=1000000)
    
    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
                    processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
                    train_interval=4, delta_clip=1.)

    dqn.compile(Adam(lr=0.00025), metrics=['mae'])

    if not loadFromExisting or args.mode == 'train':
        checkpoint_weights_file_path    = './weights/dqn_' + ENV_NAME + '_weights_{step}.h5f'
        log_filename                    = './logs/dqn_{}_log.json'.format(ENV_NAME)

        callbacks   = [ModelIntervalCheckpoint(checkpoint_weights_file_path, interval=250000)]
        callbacks   += [FileLogger(log_filename, interval=100)]

        startTime   = time.time()
        dqn.fit(env, callbacks=callbacks, nb_steps=1750000, log_interval=100000)
        endTime     = time.time()
        dqn.save_weights(file_path, overwrite=True)

    elif loadFromExisting:
        dqn.load_weights (file_path)
    
    dqn.test(env, nb_episodes=5, visualize=True)

    if not loadFromExisting or args.mode == 'train':
        print ("Time taken to train: {}".format(endTime - startTime))