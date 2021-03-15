from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ddpg import DDPGTrainer
trainer = DDPGTrainer()
trainer.restore('experiments/learning/results/save-payloadcoop-2-cc-kin-xyz_yaw-03.15.2021_12.08.55/DDPG_2021-03-15_12-09-00/DDPG_this-aviary-v0_8dcb4_00000_0_2021-03-15_12-09-00/checkpoint_1/checkpoint-1')