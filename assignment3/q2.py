import math
import gym
from frozen_lake import *
import numpy as np
import time
from utils import *
import matplotlib.pyplot as plt
import tqdm

def learn_Q_QLearning(env, num_episodes=10000, gamma = 0.99, lr = 0.1, e = 0.2, max_step=6):
	"""Learn state-action values using the Q-learning algorithm with epsilon-greedy exploration strategy(no decay)
	Feel free to reuse your assignment1's code
	Parameters
	----------
	env: gym.core.Environment
		Environment to compute Q function for. Must have nS, nA, and P as attributes.
	num_episodes: int 
		Number of episodes of training.
	gamma: float
		Discount factor. Number in range [0, 1)
	learning_rate: float
		Learning rate. Number in range [0, 1)
	e: float
		Epsilon value used in the epsilon-greedy method. 
	max_step: Int
		max number of steps in each episode

	Returns
	-------
	np.array
	  An array of shape [env.nS x env.nA] representing state-action values
	"""

	Q = np.zeros((env.nS, env.nA))
	########################################################
	#                     YOUR CODE HERE                   #
	########################################################

	avg_scores = np.zeros(num_episodes)
	total_score = 0
	for epi in range(num_episodes):
		s = env.reset()
		done = False
		t = 0
		while not done and t < max_step:
			a = Q[s].argmax()
			if Q[s].max() == 0 or np.random.uniform() < e:
				a = np.random.randint(env.nA)

			next_s, r, done, _ = env.step(a)
			q_sample = 0
			if done:
				q_sample = r
			else:
				q_sample = r + gamma * Q[next_s].max()

			Q[s, a] = (1 - lr) * Q[s, a] + lr * q_sample
			s = next_s

			total_score += r
			avg_scores[epi] = total_score / (epi + 1)

	########################################################
	#                     END YOUR CODE                    #
	########################################################
	return Q, avg_scores



def main():
	env = FrozenLakeEnv(is_slippery=False)
	plt.xlabel('episodes')
	plt.ylabel('average score')
	for e in tqdm.tqdm(range(10)):
			Q, avg_scores = learn_Q_QLearning(env, num_episodes = 10000, gamma = 0.99, lr = 0.1, e = e)
			print(Q)
			render_single_Q(env, Q)
			plt.plot(avg_scores)
	plt.legend(['m = '+str(i) for i in range(1,10,1)], loc='upper left')
	plt.show()


if __name__ == '__main__':
	main()
