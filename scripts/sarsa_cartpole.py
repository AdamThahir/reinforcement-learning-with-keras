import numpy as np
import gym
import argparse
import keras
import os
import time

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input
from keras.optimizers import Adam

from rl.agents import SARSAAgent
from rl.policy import BoltzmannQPolicy

nSteps = 500000
nHiddenLayers = 5
nHiddenLayerNodes = 25

ENV_NAME = 'CartPole-v0'
env = gym.make(ENV_NAME)

parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

if args.weights == None:
    file_path = './weights/sarsa_{0}_weights_({1}, {2}).h5f'.format(ENV_NAME, nHiddenLayers, nHiddenLayerNodes)
else:
    file_path = args.weights

loadFromExisting = True if os.path.isfile(file_path) else False

np.random.seed(2019)
env.seed(2019)
nb_actions = env.action_space.n

weight_initializer = keras.initializers.glorot_uniform()

inputLayer  = Input(shape=(1,) + env.observation_space.shape)
hiddenLayer  = Flatten(input_shape=(1,) + env.observation_space.shape)(inputLayer)
hiddenLayer = Dense (nHiddenLayerNodes, activation='relu', kernel_initializer=weight_initializer)(hiddenLayer)

if nHiddenLayers > 1:
    for i in range(1, nHiddenLayers):
        hiddenLayer = Dense(nHiddenLayerNodes, activation='relu', kernel_initializer=weight_initializer)(hiddenLayer)

outputLayer = Dense (nb_actions, activation='linear')(hiddenLayer)

model = Model(inputLayer, outputLayer)
print(model.summary())

# SARSA does not require a memory.
policy = BoltzmannQPolicy()
sarsa = SARSAAgent(model=model, nb_actions=nb_actions, nb_steps_warmup=10, policy=policy)
sarsa.compile(Adam(lr=1e-3), metrics=['mae'])

if loadFromExisting:
    sarsa.load_weights(file_path)
else:
    startTime = time.time()
    sarsa.fit(env, nb_steps=nSteps, visualize=True, verbose=1)
    endTime = time.time()
    sarsa.save_weights(file_path, overwrite=True)

# After training is done, we save the final weights.

# Finally, evaluate our algorithm for 5 episodes.
sarsa.test(env, nb_episodes=5, visualize=True)

if not loadFromExisting:
    print ("Time taken to trian: {0}".format(endTime - startTime))
