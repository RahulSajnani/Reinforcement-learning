import torch
import airsim
import numpy as np
from PIL import Image

class Drone:
    '''
    Drone class
    '''
    def __init__(self, start_position):

        self.start_position = start_position
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
        self.position.x_val = self.start_position[0]
        self.position.y_val = self.start_position[1]
        self.position.z_val = self.start_position[2]

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

