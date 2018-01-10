import gym
from gym.envs.registration import register
from kbhit import KBHit
from colorama import init
import random
import numpy as np
import matplotlib.pyplot as plt
######### for MAC OS ################
# import readchar
##############################################

# register(
# 	id='FrozenLake-v0',
# 	entry_point='gym.envs.toy_text:FrozenLakeEnv',
# 	kwargs={'map_name':'4x4', 'is_slippery':True}
# )

env = gym.make ("FrozenLake-v0")
Q = np.zeros([env.observation_space.n,env.action_space.n])
num_episodes = 5000
dis = .99
learning_rate = 0.85

rList = []
# eList = []
for i in range(num_episodes):
	# print (i)
	state = env.reset()
	done = False
	rAll = 0
	e = 1. / ((i // 100) + 1)
	# eList.append(e)
	while not done:
		# Exploit & Exploration
		# action = rargmax(Q[state,:])
		# 1. e-greedy
		if np.random.rand(1) < e:
			action = env.action_space.sample()
		else:
			action = np.argmax(Q[state,:])
		# 2. add random noise
		# action = np.argmax(Q[state,:] + np.random.randn(1, env.action_space.n)/(i+1))
		
		new_state, reward, done, _ = env.step(action)
		
		# Discounted Reward
		Q[state, action] =  (1-learning_rate) * Q[state, action] + learning_rate * (reward + dis * np.max(Q[new_state,:]))

		rAll += reward
		state = new_state
	rList.append(rAll)

print ("Success rates : " + str(sum(rList)/num_episodes))
print (Q)
# print(eList)

plt.bar(range(len(rList)), rList, color="blue")
plt.show()
#############################################
# init(autoreset=True)

# env = gym.make ("FrozenLake-v0")
# env.render()

# key = KBHit()
# env.reset()
# ########## for MAC OS ################
# # LEFT = 0
# # RIGHT = 2
# # DOWN = 1
# # UP = 3

# # arrow_keys = {
# # 	'\x1b[A' : UP,
# # 	'\x1b[B' : DOWN,
# # 	'\x1b[C' : RIGHT,
# # 	'\x1b[D' : LEFT
# # }

# while True:
# 	action = key.getarrow()
# 	########## MAC OS ################
# 	# key = readchar.readkey()
# 	# if key not in arrow_keys.keys():
# 	# 	print ("Game Over!")
# 	# 	break
# 	# action = arrow_keys[key]

# 	if action not in [0,1,2,3]:
# 		print ("Game Over!")
# 		break

# 	state, reward, done, _ = env.step(action)
# 	env.render()
# 	print ("State:",state,"Action:",action,"Done:",done)

# 	if done:
# 		print ("Finished with reward :", reward)
# 		break

#######################################
# argmax : 최대 Q를 찾아서, 그것의 action을 돌려주는 함수
# def rargmax(vector):
# 	m = np.amax(vector)
# 	indices = np.nonzero(vector==m)[0]
# 	return random.choice(indices)

# register(
# 	id='FrozenLake-v3',
# 	entry_point='gym.envs.toy_text:FrozenLakeEnv',
# 	kwargs={'map_name':'4x4', 'is_slippery':False}
# )

# env = gym.make ("FrozenLake-v3")
# Q = np.zeros([env.observation_space.n,env.action_space.n])
# num_episodes = 1000
# dis = .99

# rList = []
# # eList = []
# for i in range(num_episodes):
# 	# print (i)
# 	state = env.reset()
# 	done = False
# 	rAll = 0
# 	e = 1. / ((i // 100) + 1)
# 	# eList.append(e)
# 	while not done:
# 		# Exploit & Exploration
# 		# action = rargmax(Q[state,:])
# 		# 1. e-greedy
# 		if np.random.rand(1) < e:
# 			action = env.action_space.sample()
# 		else:
# 			action = np.argmax(Q[state,:])
# 		# 2. add random noise
# 		# action = np.argmax(Q[state,:] + np.random.randn(1, env.action_space.n)/(i+1))
		
# 		new_state, reward, done, _ = env.step(action)
		
# 		# Discounted Reward
# 		Q[state, action] = reward + dis * np.max(Q[new_state,:])

# 		rAll += reward
# 		state = new_state
# 	rList.append(rAll)

# print ("Success rates : " + str(sum(rList)/num_episodes))
# print (Q)
# # print(eList)

# plt.bar(range(len(rList)), rList, color="blue")
# plt.show()


########## for MAC OS ################
# import readchar

# init(autoreset=True)

# register(
# 	id='FrozenLake-v3',
# 	entry_point='gym.envs.toy_text:FrozenLakeEnv',
# 	kwargs={'map_name':'4x4', 'is_slippery':False}
# )

# env = gym.make ("FrozenLake-v3")
# env.render()

# # key = KBHit()

# ########## for MAC OS ################
# # LEFT = 0
# # RIGHT = 2
# # DOWN = 1
# # UP = 3

# # arrow_keys = {
# # 	'\x1b[A' : UP,
# # 	'\x1b[B' : DOWN,
# # 	'\x1b[C' : RIGHT,
# # 	'\x1b[D' : LEFT
# # }

# while True:
# 	# action = key.getarrow()
# 	########## MAC OS ################
# 	# key = readchar.readkey()
# 	# if key not in arrow_keys.keys():
# 	# 	print ("Game Over!")
# 	# 	break
# 	# action = arrow_keys[key]

# 	# if action not in [0,1,2,3]:
# 	# 	print ("Game Over!")
# 	# 	break

# 	state, reward, done, _ = env.step(action)
# 	env.render()
# 	print ("State:",state,"Action:",action,"Done:",done)

# 	if done:
# 		print ("Finished with reward :", reward)
# 		break


# import tensorflow as tf


# sess = tf.Session()
# a = tf.range(0,1000000,1)
# a = tf.reshape(a,[1000,1000])
# print(sess.run(a))

# b = tf.range(0,1000000,1)
# b = tf.reshape(b,[1000,1000])
# print(sess.run(b))

# print (sess.run(tf.matmul(a,b)))