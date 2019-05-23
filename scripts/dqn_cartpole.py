import numpy as np
import gym
import argparse
import keras
import os
import time

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

ENV_NAME = 'CartPole-v1'
nSteps = 500000
nHiddenLayers = 5
nHiddenLayerNodes = 25


parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

env = gym.make(ENV_NAME)
env.seed(2019)

if args.weights == None:
    file_path = './weights/dqn_{0}_weights_({1}, {2}).h5f'.format(ENV_NAME, nHiddenLayers, nHiddenLayerNodes)
else:
    file_path = args.weights

loadFromExisting = True if os.path.isfile(file_path) else False

# Get number of actions (this would be the nClasses for output layer)
nb_actions = env.action_space.n

print (nb_actions)
print (type(nb_actions))

# Build a simple model
weight_initializer = keras.initializers.glorot_uniform()

inputLayer  = Input(shape=(1,) + env.observation_space.shape)
hiddenLayer  = Flatten(input_shape=(1,) + env.observation_space.shape)(inputLayer)
hiddenLayer = Dense (nHiddenLayerNodes, activation='relu', kernel_initializer=weight_initializer)(hiddenLayer)

if nHiddenLayers > 1:
    for i in range(1, nHiddenLayers):
        hiddenLayer = Dense(nHiddenLayerNodes, activation='relu', kernel_initializer=weight_initializer)(hiddenLayer)

outputLayer = Dense (nb_actions, activation='linear')(hiddenLayer)

model = Model(inputLayer, outputLayer)

print (model.summary())

# Configure and compile model.

memory  = SequentialMemory(limit=nSteps, window_length=1)
policy  = BoltzmannQPolicy()

dqn     = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                    target_model_update=1e-2, policy=policy)

dqn.compile(Adam(lr=1e-3), metrics=['mae'])

if loadFromExisting:
    dqn.load_weights(file_path)
else:
    startTime = time.time()
    dqn.fit(env, nb_steps=nSteps, visualize=True, verbose=1)
    endTime = time.time()
    dqn.save_weights(file_path, overwrite=True)

dqn.test(env, nb_episodes=5, visualize=True)

if not loadFromExisting:
    print ("Time taken to trian: {0}".format(endTime - startTime))