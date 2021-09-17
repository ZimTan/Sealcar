import numpy
import time

import autocar_config
import autocar_controller.car.car_interface as car_interface
import autocar_controller.car.client as client
import autocar_controller.recorder.recorder_interface as recorder_interface
import autocar_controller.recorder.opencv_recorder as opencv_recorder

from tensorforce.environments import Environment
import MyEnvironment

from tensorforce import Agent
from tensorforce.execution import Runner

environment = Environment.create(environment='MyEnvironment', max_episode_timesteps=400)

agent = Agent.create(agent='ppo', environment=environment, batch_size=5, learning_rate=0.001)

runner = Runner( agent=agent, environment=environment, max_episode_timesteps=400)

for i in range(10):
    print("RUN FOR 20 episodes")
    runner.run(num_episodes=20)

    print("EVALUATE FOR 10 episodes")
    runner.run(num_episodes=1, evaluation=True)

print("FINISHED !")
runner.close()
