import torch
import airsim
import numpy as np
from PIL import Image

class Drone:
    '''
    Drone class
    '''
    def __init__(self, start_position, velocity_factor, hparams):

        self.start_position = start_position
        self.scaling_factor = velocity_factor
        self.reset()

    def initializeClient(self):
        '''
        Initializing airsim client 
        '''

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

    def hasCollided(self):
        '''
        Check if Drone has collided
        '''
        
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided:
            return True
        else:
            return False

    def nextAction(self, action):
        
        scaling_factor = self.scaling_factor
        
        if action.item() == 0:
            quad_offset = (scaling_factor, 0, 0)
        
        elif action.item() == 1:
            quad_offset = (0, scaling_factor, 0)
        
        elif action.item() == 2:
            quad_offset = (0, 0, scaling_factor)
        
        elif action.item() == 3:
            quad_offset = (-scaling_factor, 0, 0)
        
        elif action.item() == 4:
            quad_offset = (0, -scaling_factor, 0)
        
        elif action.item() == 5:
            quad_offset = (0, 0, -scaling_factor)

        return quad_offset

    def isDone(self, reward):
        '''
        Check if the drone has reached goal or collided
        '''

        # Either reached goal or collided
        if (reward >= self.hparams.environment.reward.goal) or (reward <= self.hparams.environment.reward.collision):
            self.reset()
            return True

    def postprocessImage(self, responses):
        '''
        Process image from airsim responses
        '''

        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
        img_rgba = img1d.reshape(response.height, response.width, 4) 
        img2d = np.flipud(img_rgba)

        image = Image.fromarray(img2d)
        image_out = np.array(image.convert("L"))

        return image_out

    def reset(self):
        '''
        Reset Drone to start position at the end of an episode
        '''

        self.initializeClient()
        self.position = self.client.simGetVehiclePose()
        
        print(self.position)
        self.position.x_val = self.start_position[0, 0]
        self.position.y_val = self.start_position[1, 0]
        self.position.z_val = self.start_position[2, 0]

        # Set init position
        # self.client.simSetVehiclePose(self.position, True, "PX4")
        self.state = self.client.getMultirotorState().kinematics_estimated.position
        print(self.state)

        # Take off with the drone
        self.client.takeoffAsync().join()
        
    def getImage(self):
        '''
        Get observation from drone sensors
        '''
        responses = self.client.simGetImages([airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])
        return self.postprocessImage(responses)


if __name__=="__main__":


    agent = Drone(5)

