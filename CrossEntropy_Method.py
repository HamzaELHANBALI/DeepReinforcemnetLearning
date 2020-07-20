import gym

import numpy as np 

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import Sequential, Softmax, CrossEntropyLoss

from recordclass import recordclass

from tensorboardX import SummaryWriter

from torch import FloatTensor, LongTensor

BATCH_SIZE = 12

percentile = 70
hidden_layer = 160


class Agent(nn.Module):
	def __init__(self,obs,act,hidden_layer):
		super(Agent,self).__init__()

		self.neuralnet = Sequential(nn.Linear(observations_size,hidden_layer), nn.ReLU(), nn.Linear(hidden_layer,actions_size))

	def forward(self,obs):
		return self.neuralnet(obs)

Episodes = recordclass('episodes','reward steps')

Episodes_Steps = recordclass('episodes_steps','observations actions')

def batch_iteration(env,NN,BATCH_SIZE):

	batch = []
	episode_rewards = 0.0
	episode_steps = []
	obs = env.reset()
	softmax = Softmax(dim=1)

	while True:

		obs_tensor = torch.FloatTensor([obs])

		actions_probability = softmax(NN(obs_tensor))

		act_proba = actions_probability.data.numpy()[0]

		action = np.random.choice(len(act_proba),p=act_proba)

		new_obs, reward, done, info = env.step(action)

		episode_rewards += reward

		episode_steps.append(Episodes_Steps(obs,action))

		if done:

			batch.append(Episodes(episode_rewards,episode_steps))
			episode_steps = []
			episode_rewards = 0.0

			new_obs = env.reset()

			if len(batch) == BATCH_SIZE:

				yield batch

				batch=[]

		obs = new_obs

def elite_batch(batch,percentile):

	rewards = list(map(lambda s: s.reward, batch))

	rewards_mean = float(np.mean(rewards))

	rewards_boundary = np.percentile(rewards,percentile)

	rewards_mean = float(np.mean(rewards))

	training_obs = []
	training_actions = []

	for example in batch:
		if example.reward < rewards_boundary:
			continue

		training_obs.extend(map(lambda step: step.observations, example.steps))
		training_actions.extend(map(lambda step: step.actions, example.steps))

	obs_tensor = FloatTensor(training_obs)
	act_tensor = LongTensor(training_actions)

	return obs_tensor, act_tensor, rewards_mean, rewards_boundary

if __name__ == "__main__":

	env = gym.make('CartPole-v1')
	env = gym.wrappers.Monitor(env,directory='monitor',force=True)
	observations_size = env.observation_space.shape[0]

	actions_size = env.action_space.n

	Neural_network = Agent(observations_size,actions_size,hidden_layer)

	loss = CrossEntropyLoss()

	optimizer = Adam(params=Neural_network.parameters(),lr=0.01)

	for iteration, batch in enumerate(batch_iteration(env,Neural_network,BATCH_SIZE)):

		train, labels, mean_rewards, boudary = elite_batch(batch,percentile)

		optimizer.zero_grad()

		scores = Neural_network(train)

		loss_value = loss(scores,labels)

		back_v = loss_value.backward()

		optimizer.step()

		print('rewards mean = ',mean_rewards)

		if mean_rewards > 500:
			print('Accomplished!')
			break

	