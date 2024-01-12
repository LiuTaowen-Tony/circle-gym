import gym
import gym_examples
import numpy as np

env = gym.make("gym_examples/CondCircle-v0", radius=1)

print(env.reset())

acceleration = np.array((0.01, -0.01))
de_acceleration = -acceleration
zero_acceleration = np.array((0,0))

print(env.step(acceleration))
print(env.step(acceleration))
print(env.step(acceleration))
print(env.step(zero_acceleration))
print(env.step(zero_acceleration))
print(env.step(zero_acceleration))
print(env.step(de_acceleration))
print(env.step(de_acceleration))
print(env.step(de_acceleration))


