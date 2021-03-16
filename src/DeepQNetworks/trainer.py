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
from collections import OrderedDict

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

        self.target_net = DQN(self.hparams.model.in_channels, self.hparams.model.actions)

        self.total_reward = 0.0
        self.episode_steps = 0.0
        self.max_episode_steps = self.hparams.model.max_episode
        self.episode_reward = 0.0
        self.populate(self.hparams.model.replay_buffer_size)


    def configure_optimizers(self):

        optimizer = getattr(torch.optim, self.hparams.optimizer.type)(self.net.parameters(), **self.hparams.optimizer.args)
        scheduler = getattr(torch.optim.lr_scheduler, self.hparams.scheduler.type)(optimizer, **self.hparams.scheduler.args)

        return [optimizer], [scheduler]

    def dqn_mse_loss(self, batch) -> torch.Tensor:
        """
        Calculates the mse loss using a mini batch from the replay buffer
        Args:
            batch: current mini batch of replay data
        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch

        #print(rewards.shape, actions.shape, "reward, action")
        # print(states["image"].shape)
        state_action_values = self.net(states["image"], states["signal"]).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # print(state_action_values)

        with torch.no_grad():
            next_state_values = self.target_net(next_states["image"], next_states["signal"]).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        #print(next_state_values.shape, "next shape")
        expected_state_action_values = next_state_values * self.hparams.model.gamma + rewards

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def populate(self, steps: int = 1000) -> None:
        '''
        Carries out several random steps through the environment to initially fill
        up the replay buffer with experiences
        '''

        for i in range(steps):
            print(i)
            self.agent.playStep(self.net, 1.0, self.get_device())

            if i % self.max_episode_steps == 0:
                self.agent.reset()

        self.agent.reset()

    def playTrajectory(self):
        '''
        Play the trajectory
        '''
        self.agent.reset()
        device = self.get_device()
        while (True):

            self.agent.playStep(self.net, 0, device)

    def training_step(self, batch, batch_idx):
        '''
        Training steps
        '''

        self.episode_steps = self.episode_steps + 1
        device = self.get_device()
        epsilon = max(self.hparams.model.min_epsilon, self.hparams.model.max_epsilon - (self.global_step + 1) / self.hparams.model.stop_decay)
        print("eps:", epsilon)

        # step through environment with agent
        reward, done = self.agent.playStep(self.target_net, epsilon, device)
        self.episode_reward += reward

        # calculates training loss
        loss = self.dqn_mse_loss(batch)

        self.log("train_loss", loss, on_epoch = True, prog_bar = True, on_step = True, logger = True)
        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0
            self.episode_steps = 0

        # Soft update of target network
        if self.global_step % self.hparams.model.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        log = {
            'total_reward': torch.tensor(self.total_reward).to(device),
            'reward': torch.tensor(reward).to(device),
            'steps': torch.tensor(self.global_step).to(device)
        }
        for key in log:
            self.log(key, log[key], logger = True, prog_bar = True, on_step = True)

        if self.episode_steps > self.max_episode_steps:
            self.episode_steps = 0
            self.total_reward = self.episode_reward
            self.agent.reset()

        #return OrderedDict({'loss': loss, 'log': log, 'progress_bar': log})
        return loss


    def __dataloader(self) -> DataLoader:
        """
        Initialize the Replay Buffer dataset used for retrieving experiences
        """

        dataset = RLDataset(self.replay_buffer, self.hparams.model.sample_size)
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

        return self.device.index if self.on_gpu else 'cpu'

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
