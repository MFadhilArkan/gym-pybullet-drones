"""Learning script for multi-agent problems.

Agents are based on `ray[rllib]`'s implementation of PPO and use a custom centralized critic.

Example
-------
To run the script, type in a terminal:

    $ python multiagent.py --num_drones <num_drones> --env <env> --obs <ObservationType> --act <ActionType> --algo <alg> --num_workers <num_workers>

Notes
-----
Check Ray's status at:
    http://127.0.0.1:8265

"""
import os
import time
import argparse
from datetime import datetime
import subprocess
import pdb
import math
import numpy as np
import pybullet as p
import pickle
import matplotlib.pyplot as plt
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Box, Dict
import torch
import torch.nn as nn
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
import ray
from ray import tune
from ray.tune.logger import DEFAULT_LOGGERS
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.env.multi_agent_env import ENV_STATE
from ray.rllib.utils.annotations import override
from ray.rllib.contrib.maddpg import MADDPGTrainer
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.multi_agent_rl.PayloadCoop import PayloadCoop
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.Logger import Logger
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer
import shared_constants
import tensorflow as tf
from ray.tune import register_env

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

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Multi-agent reinforcement learning experiments script')
    parser.add_argument('--num_drones',  default=shared_constants.NUM_DRONES,            type=int,                                                                 help='Number of drones (default: 2)', metavar='')
    parser.add_argument('--env',         default=shared_constants.ENV,      type=str,             choices=['leaderfollower', 'flock', 'meetup', 'payloadcoop'],      help='Help (default: ..)', metavar='')
    parser.add_argument('--obs',         default=shared_constants.OBS,        type=ObservationType,                                                     help='Help (default: ..)', metavar='')
    parser.add_argument('--act',         default=shared_constants.ACT,  type=ActionType,                                                          help='Help (default: ..)', metavar='')
    parser.add_argument('--algo',        default='cc',         type=str,             choices=['cc'],                                     help='Help (default: ..)', metavar='')
    parser.add_argument('--workers',     default=3,            type=int,                                                                 help='Help (default: ..)', metavar='')        
    ARGS = parser.parse_args()

    #### Save directory ########################################
    filename = os.path.dirname(os.path.abspath(__file__))+'/results/save-'+ARGS.env+'-'+str(ARGS.num_drones)+'-'+ARGS.algo+'-'+ARGS.obs.value+'-'+ARGS.act.value+'-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    #### Initialize Ray Tune ###################################
    ray.shutdown()
    ray.init(ignore_reinit_error=True)

    #### Register the environment ##############################
    temp_env_name = "this-aviary-v0" 
    register_env(temp_env_name, lambda _: PayloadCoop(num_drones=ARGS.num_drones,
                                                        aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                                                        obs=ARGS.obs,
                                                        act=ARGS.act
                                                        ))
    temp_env = PayloadCoop(num_drones=ARGS.num_drones,
                                aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                                obs=ARGS.obs,
                                act=ARGS.act,
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
        "num_workers": 0 + ARGS.workers,
        "num_envs_per_worker": 1,
        "batch_mode": "complete_episodes",

        "actor_hiddens": [64, 64],
        "critic_hiddens": [64, 64],
        "actor_lr": shared_constants.LR,
        "critic_lr": shared_constants.LR,
        "tau": 0.01,
        "train_batch_size": 32,
        "learning_starts": 32 * 1,

    }
    #### Ray Tune stopping conditions ##########################
    stop = {
        "episodes_total": 1, # 8000,
        # "episode_reward_mean": 0,
        # "training_iteration": 0,
    }
    with open(filename+'/training_constant.txt', 'w+') as f:
        for attr in dir(shared_constants):
            value = getattr(shared_constants, attr)
            if(not callable(getattr(shared_constants, attr)) and not attr.startswith("__")):
                f.write("{} : {}\n".format(attr, value))

    tr = MADDPGTrainer
    #### Train #################################################
    results = tune.run(
        tr,
        stop=stop,
        config=config,
        verbose=3,
        checkpoint_at_end=True,
        local_dir=filename,
        checkpoint_freq=50,
        # max_failures=-1,
        # restore=open("experiments/learning/results/save-payloadcoop-2-cc-kin-xyz_yaw-03.11.2021_20.57.58/checkpoint.txt").read()
        # restore = "/home/mahendra/git/gym-pybullet-drones/experiments/learning/results/save-payloadcoop-2-cc-payload_one_sensor-vel_yaw-03.12.2021_06.06.40/DDPG/DDPG_this-aviary-v0_71fdc_00000_0_2021-03-12_06-06-43/checkpoint_10/checkpoint-10"
    )
    # check_learning_achieved(results, 1.0)

    #### Save agent ############################################
    checkpoints = results.get_trial_checkpoints_paths(trial=results.get_best_trial('episode_reward_mean',
                                                                                   mode='max'
                                                                                   ),
                                                      metric='episode_reward_mean'
                                                      )
    with open(filename+'/checkpoint.txt', 'w+') as f:
        f.write(checkpoints[0][0])
    #### Shut down Ray #########################################
    ray.shutdown()
