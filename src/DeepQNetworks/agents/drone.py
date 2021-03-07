import torch
import airsim

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

    def reset(self):
        '''
        Reset Drone to start position at the end of an episode
        '''
        

        self.initializeClient()
        self.position = self.start_position
        # self.pose = self.client.simGetVehiclePose()
        self.state = self.client.getMultirotorState().kinematics_estimated.position
        print(self.state)
        
        pass


if __name__=="__main__":


    agent = Drone(5)
    
