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

        super().__init__()

    def reset(self):
        self.car.stop()
        CAR = client.UnitySimulationClient
        self.car = CAR(autocar_config.car_config)

        state = np.zeros(shape=(160 * 120,)).astype(int)
        self.node = 0
        self.old_node = 0
        self.episode_end = False
        self.finished = False
        self.change = False
        self.start_node = 0
        self.count = 0
        self.count_of_count = 0

        time.sleep(1)
            
        data = self.car.get_data()

        if 'activeNode' in data:
            self.node = data['activeNode']

        self.start_node = self.node

        return state

    def close(self):
        self.car.stop()
        super().close()


    def states(self):
        return dict(type="int", shape=(160*120,), num_values=19200)

    def actions(self):
        return {
            'steer': dict(type="int", num_values=20),
            #'throttle': dict(type="int", num_values=20)
        }

    def max_episode_timesteps(self):
        return super().max_episode_timesteps()

    def terminal(self):

        if self.node - self.start_node >= 15 :
            self.finished = True
            print("SUCEEEEEEEED !!!!!!!!!")

        #if self.node < self.old_node + 1 or self.old_node - self.node > 3:
        if self.count >= 20 or self.count_of_count >= 4:
            self.episode_end = True
            self.car.stop()
            print("Failed...")


        return self.finished or self.episode_end


    def reward(self):

        if self.finished:
            reward = 500000 #self.node - self.start_node #plus c'est long rapidement mieux c'est
            print(reward)
        elif self.change:
            if self.node < self.old_node and self.count_of_count > 5:
                reward = -1000
            else:
                reward = -1#(self.node - self.start_node) * 10
            self.change = False
        else:
            reward = -1

        reward = self.count * -1

        print("reward : ", reward, " counter : ", self.count, " counter of counter ", self.count_of_count)
        return reward

    def compute_actions(self, actions):

        time.sleep(0.2)
        action = np.array([0, 0.0])
        action[0] = (actions["steer"] - 10) / 10    
        action[1] = 0.5#(actions["throttle"]) / 20  
        self.car.send_action(action)

        self.count += 1

        data = self.car.get_data()

        if 'activeNode' in data:
            node = data['activeNode']
        else:
            node = self.node

        if node != self.node:
            self.count = 0
            self.count_of_count += 1
            self.change = True
            self.old_node = self.node
            self.node = node
        else:
            self.count_of_count = 0

        if 'image' in data:
            self.image = data['image'][:,:,0].flatten()
        return self.image


    def execute(self, actions):
        
        next_state = self.compute_actions(actions)
        terminal = self.terminal()
        reward = self.reward()
        return next_state, terminal, reward


    
