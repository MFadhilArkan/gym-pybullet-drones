"""Test script for multiagent problems.

This scripts runs the best model found by one of the executions of `multiagent.py`

Example
-------
To run the script, type in a terminal:

    $ python test_multiagent.py --exp ./results/save-<env>-<num_drones>-<algo>-<obs>-<act>-<date>

"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import numpy as np
import pybullet as p
import pickle5
import matplotlib.pyplot as plt
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Box, Dict
import torch
import torch.nn as nn
import ray
from ray import tune
from ray.tune import register_env
from ray.rllib.agents import ppo, ddpg
from ray.rllib.contrib import maddpg
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.multi_agent_rl.PayloadCoop import PayloadCoop
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync

import shared_constants

current_phase=0
class FillInActions(DefaultCallbacks):
    def on_train_result(self, trainer, result):
        global current_phase
        print("Manage Curriculum callback called on phase {}".format(current_phase))
        if result["episode_reward_mean"] > 0:
            current_phase+=1
            trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.set_phase(current_phase)))

############################################################
if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Multi-agent reinforcement learning experiments script')
    parser.add_argument('--exp',    default = "", type=str,       help='Help (default: ..)', metavar='')
    parser.add_argument('--num_drones',    default = 2, type=int,       help='Help (default: ..)', metavar='')
    ARGS = parser.parse_args()

    #### Parameters to recreate the environment ################
    NUM_DRONES =shared_constants.NUM_DRONES
    ACT = ActionType(shared_constants.ACT)
    OBS = ObservationType(shared_constants.OBS)


    #### Initialize Ray Tune ###################################
    ray.shutdown()
    ray.init(ignore_reinit_error=True, include_dashboard=True)

    #### Register the environment ##############################
    temp_env_name = "this-aviary-v0" 
    register_env(temp_env_name, lambda _: PayloadCoop(num_drones=NUM_DRONES,
                                                        aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                                                        obs=OBS,
                                                        act=ACT
                                                        ))
    temp_env = PayloadCoop(num_drones=NUM_DRONES,
                                aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                                obs=OBS,
                                act=ACT,
                                )
    action_space = temp_env.action_space[0]
    observation_space = temp_env.observation_space[0]

    obs_space_dict = {
        "agent_1": observation_space,
        "agent_2": observation_space,
    }
    act_space_dict = {
        "agent_1": action_space,
        "agent_2": action_space,
    }

    #### Config ##################################################
    config = {
        "env": temp_env_name,
        "env_config": {
            "actions_are_logits": False,
        },
        "multiagent": {
            "policies": {
                "pol0": (None, observation_space, action_space, {
                    "agent_id": 0,
                }),
                "pol1": (None, observation_space, action_space, {
                    "agent_id": 1,
                }),
            },
            "policy_mapping_fn": lambda x: "pol0" if x == 0 else "pol1",
        },
        "framework": "tf",
        "callbacks":FillInActions,
        "gamma": shared_constants.GAMMA,
        "n_step": shared_constants.N_STEP,
        "num_workers": 0 + 4,
        "num_envs_per_worker": 1,
        "batch_mode": "complete_episodes",

        "actor_hiddens": [64, 64],
        "critic_hiddens": [64, 64],
        "actor_lr": shared_constants.LR,
        "critic_lr": shared_constants.LR,
        "tau": 0.01,
        "train_batch_size": 512,
        "learning_starts": 512 * 10,
        "explore": False,
        "in_evaluation": True

    }
    #### Restore agent #########################################
    agent = maddpg.MADDPGTrainer(config=config)
    checkpoint = "experiments/learning/results/save-payloadcoop-2-cc-payload_one_sensor-xyz_yaw-03.23.2021_14.43.35/MADDPG_2021-03-23_14-43-38/MADDPG_this-aviary-v0_7ae3a_00000_0_2021-03-23_14-43-38/checkpoint_1/checkpoint-1"
    agent.restore(checkpoint)


    
    #### Create test environment ###############################
    test_env = PayloadCoop(num_drones=NUM_DRONES,
                            aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                            obs=OBS,
                            act=ACT,
                            gui=True,
                            record=True
    )
    for ep in range(10):
        #### Show, record a video, and log the model's performance #
        obs = test_env.reset()
        test_env.set_phase(12)
        logger = Logger(logging_freq_hz=int(test_env.SIM_FREQ/test_env.AGGR_PHY_STEPS),
                        num_drones=NUM_DRONES
                        )
        action = {i: action_space.sample() for i in range(NUM_DRONES)}
        start = time.time()

        policy0 = agent.get_policy("pol0")
        policy1 = agent.get_policy("pol1")

        START = time.time()
        for i in range(0, int(test_env.EPISODE_LEN_SEC * test_env.SIM_FREQ), test_env.AGGR_PHY_STEPS):
            #### Step the simulation ###################################
            temp = {}
            act0 = policy0.compute_actions(obs[0].reshape(1, -1))[0][0]
            act1 = policy1.compute_actions(obs[1].reshape(1, -1))[0][0]
            action = {0: act0, 1: act1}
            obs, reward, done, info = test_env.step(action)
            #### Printout ##############################################
            if i%test_env.SIM_FREQ == 0: #setiap 1 detik
                test_env.render()
            #### Sync the simulation ###################################

            if test_env.GUI:
                sync(i, START, test_env.TIMESTEP)
            
            if(done['__all__']):  
                break
        


    #### Shut down Ray #########################################
    ray.shutdown()