from gym.envs.registration import register

register(
    id="gym_examples/CondCircle-v0",
    entry_point="gym_examples.envs:CondCircleEnv",
    max_episode_steps=300
)
