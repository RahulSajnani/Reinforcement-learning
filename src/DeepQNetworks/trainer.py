import torch
from models.DQN import DQN
from sensors.heat_sensor import HeatSensor


class AgentTrainer:

    def __init__(self, agent, sensor, source_position, hparams):
        
        self.agent = agent
        self.heat_sensor = sensor
        self.source_position = source_position
        self.hparams = hparams

    def train(self):
        pass



if __name__=="__main__":

    source_position = torch.tensor([[1], [2], [3]]).float()
    agent_position = torch.tensor([[4], [2], [3]]).float()
    heat_sensor = HeatSensor(source_position)
    
    trainer = AgentTrainer(0, heat_sensor, source_position, 0)
