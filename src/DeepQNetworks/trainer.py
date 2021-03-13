import torch
from models.DQN import DQN
from dataset.RLdataset import *
from sensors.heat_sensor import HeatSensor
from agents.drone import Drone
import os, hydra, logging, glob
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger
from torch.utils.data import DataLoader
import torch.nn as nn

class AgentTrainer(pl.LightningModule):
    '''
    Pytorch trainer class for Drone Reinforcement learning
    '''

    def __init__(self, hparams):
        '''
        Initializations
        '''
        super().__init__()
        self.hparams = hparams

        # Position of human
        source_position = torch.tensor([[self.hparams.environment.position.end.x],
                                        [self.hparams.environment.position.end.y],
                                        [self.hparams.environment.position.end.z]]).float()

        # Position of agent
        agent_position  = torch.tensor([[self.hparams.environment.position.start.x],
                                        [self.hparams.environment.position.start.y],
                                        [self.hparams.environment.position.start.z]]).float()


        # Initialize sensor
        self.heat_sensor = HeatSensor(source_position,
                                      strength_factor = self.hparams.environment.sensor.signal_strength_factor,
                                      reward_factor = self.hparams.environment.reward.factor)


        # Initialize Replay buffer
        self.replay_buffer = ReplayBuffer(capacity = self.hparams.model.replay_buffer_size)


        # Initialize drone
        self.agent = Drone(start_position = agent_position,
                           velocity_factor = self.hparams.environment.agent.velocity_factor,
                           hparams = self.hparams,
                           buffer = self.replay_buffer,
                           sensor = self.heat_sensor)

        self.net = DQN(self.hparams.model.in_channels, self.hparams.model.actions)

        self.total_reward = 0.0
        self.episode_reward = 0.0
        self.populate()


    def configure_optimizers(self):

        optimizer = getattr(torch.optim, self.hparams.optimizer.type)(self.net.parameters())
        scheduler = getattr(torch.optim.lr_scheduler, self.hparams.scheduler.type)(optimizer, **self.hparams.scheduler.args)

        return [optimizer], [scheduler]

    def populate(self, steps: int = 1000) -> None:
        """
        Carries out several random steps through the environment to initially fill
        up the replay buffer with experiences
        Args:
            steps: number of random steps to populate the buffer with
        """

        for i in range(steps):
            print(i)
            self.agent.playStep(self.net, self.get_device(), populate = 1)


    def training_step(self, batch, batch_idx):

        pass


    def __dataloader(self) -> DataLoader:
        """
        Initialize the Replay Buffer dataset used for retrieving experiences
        """

        dataset = RLDataset(self.buffer, self.episode_length)
        dataloader = DataLoader(
            dataset=dataset,
            **self.hparams.dataset.loader)

        return dataloader

    def train_dataloader(self) -> DataLoader:
        """
        Get train loader
        """

        return self.__dataloader()

    def get_device(self) -> str:
        """
        Retrieve device currently being used by minibatch
        """

        return self.net.device.index if self.on_gpu else 'cpu'

    def forward(self, x):

        return self.net(x)


seed_everything(123)
log = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="DQN.yaml")
def main(cfg):

    tb_logger = TensorBoardLogger(save_dir = "./")
    log.info(cfg.pretty())
    model = AgentTrainer(hparams = cfg)
    trainer = Trainer(**cfg.trainer, logger = tb_logger)
    trainer.fit(model)


if __name__=="__main__":

    main()
