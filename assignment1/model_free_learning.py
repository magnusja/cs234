### Episode model free learning using Q-learning and SARSA

# Do not change the arguments and output types of any of the functions provided! You may debug in Main and elsewhere.

import numpy as np
import gym
import time
from lake_envs import *
import matplotlib.pyplot as plt
import tqdm

def learn_Q_QLearning(env, num_episodes=2000, gamma=0.95, lr=0.1, e=0.8, decay_rate=0.99):
  """Learn state-action values using the Q-learning algorithm with epsilon-greedy exploration strategy.
  Update Q at the end of every episode.

  Parameters
  ----------
  env: gym.core.Environment
    Environment to compute Q function for. Must have nS, nA, and P as
    attributes.
  num_episodes: int 
    Number of episodes of training.
  gamma: float
    Discount factor. Number in range [0, 1)
  learning_rate: float
    Learning rate. Number in range [0, 1)
  e: float
    Epsilon value used in the epsilon-greedy method. 
  decay_rate: float
    Rate at which epsilon falls. Number in range [0, 1)

  Returns
  -------
  np.array
    An array of shape [env.nS x env.nA] representing state, action values
  """
  
  ############################
  # YOUR IMPLEMENTATION HERE #
  ############################
  Q = np.zeros((env.nS, env.nA))

  for epi in range(num_episodes):
    s = env.reset()
    done = False
    while not done:
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
    
    if epi % 50 == 0:
      e = e * decay_rate

  return Q

def learn_Q_SARSA(env, num_episodes=2000, gamma=0.95, lr=0.1, e=0.8, decay_rate=0.99):
  """Learn state-action values using the SARSA algorithm with epsilon-greedy exploration strategy
  Update Q at the end of every episode.

  Parameters
  ----------
  env: gym.core.Environment
    Environment to compute Q function for. Must have nS, nA, and P as
    attributes.
  num_episodes: int 
    Number of episodes of training.
  gamma: float
    Discount factor. Number in range [0, 1)
  learning_rate: float
    Learning rate. Number in range [0, 1)
  e: float
    Epsilon value used in the epsilon-greedy method. 
  decay_rate: float
    Rate at which epsilon falls. Number in range [0, 1)

  Returns
  -------
  np.array
    An array of shape [env.nS x env.nA] representing state-action values
  """

  ############################
  # YOUR IMPLEMENTATION HERE #
  ############################
  Q = np.zeros((env.nS, env.nA))
  for epi in range(num_episodes):
    s = env.reset()
    done = False
    while not done:
      a = Q[s].argmax()
      if Q[s].max() == 0 or np.random.uniform() < e:
        a = np.random.randint(env.nA)

      next_s, r, done, _ = env.step(a)
      a_ = Q[next_s].argmax()
      if np.random.uniform() < e:
        a_ = np.random.randint(env.nA)
      q_sample = 0
      if done:
        q_sample = r
      else:
        q_sample = r + gamma * Q[next_s, a_]

      Q[s, a] = (1 - lr) * Q[s, a] + lr * q_sample
      s = next_s
    
    if epi % 50 == 0:
      e = e * decay_rate

  return Q

def render_single_Q(env, Q):
  """Renders Q function once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play Q function on. Must have nS, nA, and P as
      attributes.
    Q: np.array of shape [env.nS x env.nA]
      state-action values.
  """

  episode_reward = 0
  state = env.reset()
  done = False
  while not done:
    env.render()
    time.sleep(0.5) # Seconds between frames. Modify as you wish.
    action = np.argmax(Q[state])
    state, reward, done, _ = env.step(action)
    episode_reward += reward

  print("Episode reward: %f" % episode_reward)

def evaluate_q(env, Q):
  episode_reward = 0
  state = env.reset()
  done = False
  i = 0
  while not done and i < 500:
    action = np.argmax(Q[state])
    state, reward, done, _ = env.step(action)
    episode_reward += reward
  return episode_reward

def avg_score(env, q, num_repeats=1000):
  return np.mean([evaluate_q(env, q) for _ in range(num_repeats)])

def learn_and_evaluate_q(env, q_func, episodes):
  result = np.zeros_like(episodes).astype(float)
  for i in tqdm.tqdm(range(episodes.shape[0])):
    q = q_func(env, episodes[i])
    result[i] = avg_score(env, q)
  return result

# Feel free to run your own debug code in main!
def main():
  env = gym.make('Stochastic-4x4-FrozenLake-v0')
  #Q = learn_Q_QLearning(env)
  #Q = learn_Q_SARSA(env)
  #render_single_Q(env, Q)

  episodes = np.arange(0, 5000, 10)
  q_learning = learn_and_evaluate_q(env, learn_Q_QLearning, episodes)
  sarsa = learn_and_evaluate_q(env, learn_Q_QLearning, episodes)

  plt.plot(episodes, q_learning)
  plt.plot(episodes, sarsa)

  print(q_learning[-1])
  print(sarsa[-1])

  plt.legend(['q-learning', 'sarsa'], loc='upper left')

  plt.show()

if __name__ == '__main__':
    main()
