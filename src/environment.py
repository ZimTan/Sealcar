from tensorforce.environments import Environment
import autocar_controller.car.client as client


class CustomEnvironment(Environment):

    def __init__(self):
        self.car = client.UnitySimulationClient(autocar_config.car_config)
        super().__init__()

    def close(self):
        self.car.stop()


    def states(self):
        return {"image": np.array}

    def actions(self):
        return {
            'steer': dict(type="int", num_values=20),
            'throttle': dict(type="int", num_values=20)
        }

    def max_episode_timesteps(self):
        return 100

    def terminal(self):
        self.finished = False #succeed ?
        self.episode_end = False #not succeed ?
        return self.finished or self.episode_end


    def reward(self):

        if self.finished:
            reward = self.node #plus c'est long rapidement mieux c'est
        else:
            reward = -1

        return reward

    def compute_actions(self, actions):

        action = numpy.array([0, 0.0])
        action[0] = actions["steer"]        #HardCode_steer[counter]#controller.get_steer()
        action[1] = actions["throttle"]     #HardCode_throttle[counter]#controller.get_throttle()
        car.send_action(action)

        data = car.get_data()

        self.node = data['activeNode']

        return data['image']


    def execute(self, actions):
        
        next_state = self.compute_actions(actions)
        terminal = self.terminal()
        reward = self.reward()
        return next_state, terminal, reward


    
