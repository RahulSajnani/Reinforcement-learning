import torch


'''
Author: Rahul Sajnani
Date  : 1 March 2021
'''

class HeatSensor:
    '''
    Dummy sensor that emulates the heat signature of human
    '''

    def __init__(self, source_position, strength_factor = 0.002, reward_factor = 0.001):
        '''
        Input:
            source_position - torch.Tensor 3 x 1 - x, y, z - Position of human
        '''

        self.source_position = source_position
        self.strength_factor = strength_factor
        self.reward_factor   = reward_factor
        self.max_distance = 2500
        self.cos = torch.nn.CosineSimilarity()

    def getSignalStrength(self, position):
        '''
        Function to obtain the signal strength of the human given the location of the RL agent

        Input:
            position - torch.Tensor 3 x 1 - x, y, z -
        '''

        distance = self.getDistanceFromDestination(position)
        signal = torch.tensor([[ 1 / (self.strength_factor * distance + 0.1)]])

        print(signal, "signal")
        return signal

    def getDistanceFromDestination(self, position):
        '''
        Get distance from source of the signal
        '''

        distance = torch.sqrt(torch.sum(torch.square(position - self.source_position)))

        return distance

    def getReward(self, position, start_position, velocity):
        '''
        Get the reward for being reaching a position
        '''

        distance = self.getDistanceFromDestination(position)
        distance_start = self.getDistanceFromDestination(start_position)

        direction = self.source_position - position
        #print(direction.shape, velocity.shape)
        # reward   = self.reward_factor * (- distance)
        reward = - (distance / self.max_distance) + 0.5 * self.cos(direction.squeeze().unsqueeze(0), velocity.unsqueeze(0))  #- 0.00001 * torch.norm(velocity)*( self.max_distance / (distance + 100))

        reward   = torch.tensor([reward])

        return reward


if __name__ == "__main__":


    source_position = torch.tensor([[1], [2], [3]]).float()
    agent_position = torch.tensor([[4], [2], [3]]).float()
    sensor = HeatSensor(source_position)
    r_t = sensor.getReward(agent_position)
    signal_t = sensor.getSignalStrength(agent_position)
    print(r_t, signal_t)
