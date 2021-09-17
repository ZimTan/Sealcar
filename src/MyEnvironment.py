

from tensorforce.environments import Environment
import autocar_controller.car.client as client
import autocar_config
import numpy as np
import time
import matplotlib.pyplot as plt

def np_rgb2l(img):

    maxc = np.max(img, -1)
    minc = np.min(img, -1)
    l = (minc + maxc) / 2.0

    return l

def close_event():
    plt.close()

def np_rgb2l2(img):

    return (0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2])


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
        print("START ", self.start_z)

        if 'image' in data:
            self.img = self.segmentation2(data['image'])


        self.episode_end = False
        self.finished = False
        self.throttle = 0
            
        return self.img

    def close(self):
        self.car.stop()
        super().close()


    def states(self):
        return dict(type="float", shape=(24, 54))

    def actions(self):
        return {
            'steer': dict(type="int", num_values=11),
            'throttle': dict(type="int", num_values=11)
        }

    def max_episode_timesteps(self):
        return super().max_episode_timesteps()

    def terminal(self):

        self.episode_end = self.img_line.mean() > 10

        if self.episode_end:
            print("EPISODE END :", self.img_line.mean())

        """
        fig, axs = plt.subplots(ncols=2)
        timer = fig.canvas.new_timer(interval = 200)
        timer.add_callback(close_event)

        axs[0].imshow(self.img * 255, cmap='gray', vmin=0, vmax=255)
        if (img_line.mean() >= 1):
            print(img_line.mean())
        axs[1].imshow(img_line, cmap='gray', vmin=0, vmax=255)
        

        timer.start()
        plt.show()
        if self.finished :
            print("FINISHHHHHH ! ")
        """

        return self.finished or self.episode_end


    def reward(self):

        if self.episode_end:
            reward = -10
        else:
            reward = self.throttle

        return reward

    def segmentation(self, img):

        fig, axs = plt.subplots(ncols=3)
        timer = fig.canvas.new_timer(interval = 200)
        timer.add_callback(close_event)

        axs[0].imshow(img)
        l = np_rgb2l2(img)

        l[:50,:] = 0
        l[l < 210] = 0
        l[l > 220] = 255
        axs[1].imshow(l)

        axs[2].imshow(l[::4,::4])
        

        timer.start()
        plt.show()

        return l

    
    def segmentation2(self, img):

        THRESHOLD = 210
        HORIZON = 50
        #fig, axs = plt.subplots(ncols=3)
        #timer = fig.canvas.new_timer(interval = 200)
        #timer.add_callback(close_event)

        img_grey = 0.299 * img[:,:,0] +  0.587 * img[:,:,2] +  0.114 * img[:,:,2] 
        #axs[0].imshow(img_grey, cmap='gray', vmin=0, vmax=255)

        img_grey[:HORIZON,:] = 0

        rows, cols = img_grey.shape

        for i in range(rows - 1, HORIZON - 1, -1):
            mask = img_grey[i,:] > THRESHOLD
            if mask.sum() > 20:
                img_grey[i,:] = 0


        for i in range(rows - 1, HORIZON - 1, -1):

            found = False

            j = cols // 2
            while j < cols - 1:

                if (found == False and img_grey[i,j] > THRESHOLD) : #white

                    found = True
                    
                    if j < cols - 1:
                        j += 1

                    while (img_grey[i,j] > THRESHOLD - 30) :

                        if j < cols - 1 :
                            j += 1
                        else:
                            break

                else:
                    img_grey[i,j] = 0

                j += 1



            found = False
            j = cols // 2
            while j > 0:

                if (found == False and img_grey[i,j] > THRESHOLD) : #white

                    found = True

                    if j == 0 :
                        continue

                    j -= 1


                    while (img_grey[i,j] > THRESHOLD - 30) :

                        if j > 0 :
                            j -= 1
                        else:
                            break

                else:
                    img_grey[i,j] = 0
                j -= 1

        #axs[1].imshow(img_grey, cmap='gray', vmin=0, vmax=255)
        img_seg = img_grey[50:,:][::3,::3]
        #axs[2].imshow(img_seg, cmap='gray', vmin=0, vmax=255)

        #timer.start()
        #plt.show()

        return img_seg
            

    def compute_actions(self, actions):

        action = np.array([0, 0.0])
        action[0] = (actions["steer"] - 5) / 5
        action[1] = (actions["throttle"]) / 10
        self.car.send_action(action)

        time.sleep(0.01)

        self.throttle = action[1]

        self.count += 1

        data = self.car.get_data()

        if 'image' in data:
            self.img = self.segmentation2(data['image'])
            self.img_line = self.img[20:, 18:32]


        return self.img


    def execute(self, actions):
        
        next_state = self.compute_actions(actions)
        terminal = self.terminal()
        reward = self.reward()
        return next_state, terminal, reward


    
