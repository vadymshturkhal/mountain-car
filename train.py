import gymnasium as gym


# Create the environment
env = gym.make('MountainCar-v0', render_mode="human")
observation = env.reset()

done = False
while not done:
    env.render()

env.close()
