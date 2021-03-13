from torch.serialization import storage_to_tensor_type
from sensors.heat_sensor import HeatSensor
import torch
import airsim
from collections import deque, namedtuple
import numpy as np
import time
from PIL import Image

Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class Drone:
    '''
    Drone class
    '''
    def __init__(self, start_position, velocity_factor, hparams, buffer, sensor):
        '''
        Drone initializations
        start position, velocity factor, hyper parameters, replay buffer
        '''
        self.buffer = buffer
        self.sensor = sensor
        self.hparams = hparams
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

    def hasReachedGoal(self):
        '''
        Check if Drone has reached the goal (distance is less than the velocity factor)
        '''
        
        current_position = self.convertPositionToTensor(self.position.position)
        distance = self.sensor.getDistanceFromDestination(current_position)
        if distance < self.scaling_factor:
            return True
        else:
            return False

    def nextAction(self, action):
        '''
        Get change of position from action index
        '''

        scaling_factor = self.scaling_factor
        if action == 0:
            quad_offset = (scaling_factor, 0, 0)

        elif action == 1:
            quad_offset = (0, scaling_factor, 0)

        elif action == 2:
            quad_offset = (0, 0, scaling_factor)

        elif action == 3:
            quad_offset = (-scaling_factor, 0, 0)

        elif action == 4:
            quad_offset = (0, -scaling_factor, 0)

        elif action == 5:
            quad_offset = (0, 0, -scaling_factor)

        return quad_offset
    

    def convertPositionToTensor(self, position):
        '''
        Converts position from airsim vector 3d to 3 x 1 tensor
        '''

        current_position = torch.tensor([[position.x_val], [position.y_val], [position.z_val]])

        return current_position


    def getAgentState(self):
        '''
        Get agent state (Image and signal strength)
        '''

        position = self.convertPositionToTensor(self.position)
        state_image = self.getImage()
        state_signal_strength = self.senor.getReward(position)
        
        state_image = torch.tensor(state_image).unsqueeze(0)
        state_signal_strength = torch.tensor([state_signal_strength]).unsqueeze(0)

        return {"image": state_image, "signal": state_signal_strength}

    def getAction(self, net, device):
        '''
        Perform action
        '''

        if np.random.random() < self.hparams.agent.epsilon:
            action = np.random.randint(self.hparams.model.actions)
        else:
            state_dict = self.getAgentState()
            
            if device not in ['cpu']:
                for key in state_dict:
                    state_dict[key] = state_dict[key].cuda(device)
            
            q_values = net(state_dict["image"], state_dict["signal"])
            _, action = torch.max(q_values, dim = 1)
            action = int(action.item())

        return action
        

    @torch.no_grad()
    def playStep(self, net, device):
        '''
        Performs one step in the environment
        
        Input:
            net - DQN network
            device - device 
        
        Returns:
            reward - instantaneous reward
            done - Check if episode is done
        '''
        
        
        action = self.getAction()
        action_offset = self.nextAction(action)
        quad_state = self.client.getMultirotorState().kinematics_estimated.position
        
        state_dict = self.getAgentState()

        self.client.moveToPositionAsync(
            quad_state.x_val + action_offset[0],
            quad_state.y_val + action_offset[1],
            quad_state.z_val + action_offset[2],
            20,
        ).join()
        time.sleep(0.5)

        current_position = self.convertPositionToTensor(self.position.position)
        done, reward = self.isDone()
        
        new_state_dict = self.getAgentState()
        self.position = self.client.simGetVehiclePose()

        if not done:
            reward = self.sensor.getReward(current_position)

        exp = Experience(state_dict, action, reward, done, new_state_dict)
        self.buffer.append(exp)

        if done:
            self.reset()
        
        return reward, done

    def isDone(self):
        '''
        Check if the drone has reached goal or collided
        '''

        reward = 0

        # Either reached goal or collided
        if self.hasReachedGoal():
            reward = self.hparams.environment.reward.goal
            done = True
        elif self.hasCollided():
            reward = self.hparams.environment.reward.collision
            done = True
        else:
            done = False

        return done, reward


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

        pose_params = self.hparams.environment
        start = self.start_position.numpy()

        # Set initial pose 
        self.position.position.x_val = float(start[0, 0])
        self.position.position.y_val = float(start[1, 0])
        self.position.position.z_val = float(start[2, 0])

        self.position.orientation.w_val = pose_params.quaternion.start.w_val
        self.position.orientation.x_val = pose_params.quaternion.start.x_val
        self.position.orientation.y_val = pose_params.quaternion.start.y_val
        self.position.orientation.z_val = pose_params.quaternion.start.z_val

        # print("position, ", self.position)
        # Set init position
        self.client.simSetVehiclePose(self.position, True)
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

