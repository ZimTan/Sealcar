from tensorforce.environments import Environment
import autocar_controller.car.client as client
import autocar_config
import numpy as np
import time

class CustomEnvironment(Environment):

    
    CAR = client.UnitySimulationClient


    def __init__(self):

        CAR = client.UnitySimulationClient
        self.car = CAR(autocar_config.car_config)

        data = self.car.get_data()

        self.start_z = data['pos_z']
        self.pos_z = 0
        self.count = 0

        super().__init__()

    def reset(self):
        self.car.stop()
        CAR = client.UnitySimulationClient
        self.car = CAR(autocar_config.car_config)

        data = self.car.get_data()
        self.start_z = data['pos_z']
        print("START ", self.start_z)
        self.count = 0
        self.pos_z = 0

        self.episode_end = False
        self.finished = False
            
        return self.pos_z

    def close(self):
        self.car.stop()
        super().close()


    def states(self):
        return dict(type="float")

    def actions(self):
        return {
            #'steer': dict(type="int", num_values=20),
            'throttle': dict(type="int", num_values=21)
        }

    def max_episode_timesteps(self):
        return super().max_episode_timesteps()

    def terminal(self):

        self.finished = self.pos_z > 10
        if self.finished :
            print("FINISHHHHHH ! ")

        return self.finished or self.episode_end


    def reward(self):

        if self.finished:
            reward =  np.log(((300000 - self.count) ** 2)) #self.node - self.start_node #plus c'est long rapidement mieux c'est
            print(reward)
        else:
            reward = -1

        return reward

    def compute_actions(self, actions):

        action = np.array([0, 0.0])
        action[0] = 0 #(actions["steer"] - 10) / 10    
        action[1] = (actions["throttle"] - 10) / 10  
        self.car.send_action(action)

        self.count += 1

        data = self.car.get_data()
        if 'pos_z' in data:
            self.pos_z = data['pos_z'] - self.start_z

        print(self.pos_z)

        return self.pos_z


    def execute(self, actions):
        
        next_state = self.compute_actions(actions)
        terminal = self.terminal()
        reward = self.reward()
        return next_state, terminal, reward


    
