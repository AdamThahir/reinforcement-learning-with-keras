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


ENV_NAME = 'CartPole-v0'
nSteps = 500000
nHiddenLayers = 5
nHiddenLayerNodes = 25

parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(2019)
env.seed(2019)

nb_actions = env.action_space.n

if args.weights == None:
    file_path = './weights/duel_dqn_{0}_weights_({1}, {2}).h5f'.format(ENV_NAME, nHiddenLayers, nHiddenLayerNodes)
else:
    file_path = args.weights

loadFromExisting = True if os.path.isfile(file_path) else False

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

memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()

# you can specify the dueling_type to one of {'avg','max','naive'}
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)
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
