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
from ray.rllib.agents.ppo import PPOTrainer, PPOTFPolicy
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.multi_agent_rl.PayloadCoop import PayloadCoop
from gym_pybullet_drones.envs.multi_agent_rl.FlockAviary import FlockAviary
from gym_pybullet_drones.envs.multi_agent_rl.LeaderFollowerAviary import LeaderFollowerAviary
from gym_pybullet_drones.envs.multi_agent_rl.MeetupAviary import MeetupAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override

import shared_constants
action_model = nn.Sequential(
            nn.Linear(8 + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
        )

value_model = nn.Sequential(
    nn.Linear(21, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
)

class CustomTorchCentralizedCriticModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.action_model = action_model
        self.value_model = value_model
        self._model_in = None

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._model_in = [input_dict["obs_flat"], state, seq_lens]
        obs = input_dict["obs"]["own_obs"].float()
        opp_obs = input_dict["obs"]["opponent_obs"].float()
        obs = torch.cat((obs,opp_obs[:, [0,1]]), dim = 1) #tambah sensor dan yaw drone lain untuk observasi
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._features = self.action_model(self._last_flat_in)
        return self._features, state

    @override(ModelV2)
    def value_function(self):
        value_out, _ = self.value_model({"obs": self._model_in[0]}, self._model_in[1], self._model_in[2])
        return torch.reshape(value_out, [-1])

############################################################
current_phase=0
class FillInActions(DefaultCallbacks):
    def on_postprocess_trajectory(self, worker, episode, agent_id, policy_id, policies, postprocessed_batch, original_batches, **kwargs):
        to_update = postprocessed_batch[SampleBatch.CUR_OBS]
        other_id = 1 if agent_id == 0 else 0
        action_encoder = ModelCatalog.get_preprocessor_for_space( 
                                                                 Box(-np.inf, np.inf, (ACTION_VEC_SIZE,), np.float32) # Unbounded
                                                                 )
        # action_encoder = ModelCatalog.get_preprocessor_for_space( 
        #                                                          Discrete(ACTION_VEC_SIZE) # Unbounded
        #                                                          )
        _, opponent_batch = original_batches[other_id]
        opponent_actions = np.array([action_encoder.transform(a) for a in opponent_batch[SampleBatch.ACTIONS]])
        to_update[:, -ACTION_VEC_SIZE:] = opponent_actions

    def on_train_result(self, trainer, result):
        global current_phase
        print("Manage Curriculum callback called on phase {}".format(current_phase))
        if result["episode_reward_mean"] > 0:
            current_phase+=1
            trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.set_phase(current_phase)))


############################################################
def central_critic_observer(agent_obs, **kw):
    new_obs = {
        0: {
            "own_obs": agent_obs[0],
            "opponent_obs": agent_obs[1],
            "opponent_action": np.zeros(ACTION_VEC_SIZE), # Filled in by FillInActions
        },
        1: {
            "own_obs": agent_obs[1],
            "opponent_obs": agent_obs[0],
            "opponent_action": np.zeros(ACTION_VEC_SIZE), # Filled in by FillInActions
        },
    }
    return new_obs
############################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-agent reinforcement learning experiments script')
    parser.add_argument('--num_drones',  default=shared_constants.NUM_DRONES,            type=int,                                                                 help='Number of drones (default: 2)', metavar='')
    parser.add_argument('--env',         default=shared_constants.ENV,      type=str,             choices=['leaderfollower', 'flock', 'meetup', 'payloadcoop'],      help='Help (default: ..)', metavar='')
    parser.add_argument('--obs',         default=shared_constants.OBS,        type=ObservationType,                                                     help='Help (default: ..)', metavar='')
    parser.add_argument('--act',         default=shared_constants.ACT,  type=ActionType,                                                          help='Help (default: ..)', metavar='')
    parser.add_argument('--algo',        default='cc',         type=str,             choices=['cc'],                                     help='Help (default: ..)', metavar='')
    parser.add_argument('--workers',     default=7,            type=int,                                                                 help='Help (default: ..)', metavar='')        
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
    observer_space = Dict({
        "own_obs": temp_env.observation_space[0],
        "opponent_obs": temp_env.observation_space[0],
        "opponent_action": temp_env.action_space[0],
    })
    action_space = temp_env.action_space[0]
    OWN_OBS_VEC_SIZE = observer_space["own_obs"].shape[0]
    ACTION_VEC_SIZE = action_space.shape[0]
    
    ModelCatalog.register_custom_model("cc_model", CustomTorchCentralizedCriticModel)
    config = {
        "env": temp_env_name,
        "num_workers":1,
        "num_envs_per_worker": 1,
        # "num_gpus": 0,
        # "batch_mode": "complete_episodes",
        "callbacks":FillInActions,
        # "framework": "torch",
        # "lr" : shared_constants.LR,
        # "gamma": shared_constants.GAMMA,
        # "n_step": shared_constants.N_STEP,
        # "tau": 0.005,
        # "train_batch_size": 128,
        # "learning_starts": 128 * 1,
        # # "batch_mode": "complete_episodes",
        # # "train_batch_size":200,
        # # "rollout_fragment_length":200,
        # "exploration_config": {
        #     "type": "OrnsteinUhlenbeckNoise",
        #     "ou_base_scale": 0.1,
        #     # The OU theta param.
        #     "ou_theta": 0.15,
        #     # The OU sigma param.
        #     "ou_sigma": 0.2,
        # },
        "explore": False
    }
    # config["model"] = { 
    #     "custom_model": "cc_model",
    # }
    config["multiagent"] = { 
        "policies": {
            # "pol0": (None, observer_space, action_space, {"agent_id": 0,}),
            # "pol1": (None, observer_space, action_space, {"agent_id": 1,}),
            "pol0": (None, temp_env.observation_space[0], temp_env.action_space[0], {"agent_id": 0,}),
            "pol1": (None, temp_env.observation_space[1], temp_env.action_space[1], {"agent_id": 1,}),
        },
        "policy_mapping_fn": lambda x: "pol0" if x == 0 else "pol1", # # Function mapping agent ids to policy ids
        # "observation_fn": central_critic_observer, # See rllib/evaluation/observation_function.py for more info
    }


    #### Restore agent #########################################
    agent = ddpg.DDPGTrainer(config=config)
    # with open(ARGS.exp+'/checkpoint.txt', 'r+') as f:
    #     checkpoint = f.read()
    checkpoint = "/home/mahendra/git/gym-pybullet-drones/experiments/learning/results/save-payloadcoop-2-cc-payload_one_sensor-xyz_yaw-03.25.2021_19.59.31/DDPG_2021-03-25_19-59-34/DDPG_this-aviary-v0_f271e_00000_0_2021-03-25_19-59-34/checkpoint_10/checkpoint-10"
    agent.restore(checkpoint)

    #### Extract and print policies ############################
    policy0 = agent.get_policy("pol0")
    # print("action model 0", policy0.model.action_model)
    # print("value model 0", policy0.model.value_model)
    policy1 = agent.get_policy("pol1")
    # print("action model 1", policy1.model.action_model)
    # print("value model 1", policy1.model.value_model)

    #### Create test environment ###############################
    test_env = PayloadCoop(num_drones=NUM_DRONES,
                            aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                            obs=OBS,
                            act=ACT,
                            gui=True
    )
    #### Show, record a video, and log the model's performance #
    
    for ep in range(10):
        obs = test_env.reset()
        logger = Logger(logging_freq_hz=int(test_env.SIM_FREQ/test_env.AGGR_PHY_STEPS),
                        num_drones=NUM_DRONES
                        )
        test_env.set_phase(12)
        action = {i: action_space.sample() for i in range(NUM_DRONES)}
        START = time.time()
        return_ = 0
        for i in range(0, int(test_env.EPISODE_LEN_SEC * test_env.SIM_FREQ), test_env.AGGR_PHY_STEPS): # Up to 6''
            #### Deploy the policies ###################################
            temp = {}
            temp[0] = policy0.compute_single_action(obs[0]) 
            temp[1] = policy1.compute_single_action(obs[1])
            action = {0: temp[0][0], 1: temp[1][0]}
            obs, reward, done, info = test_env.step(action)
            if i%test_env.SIM_FREQ == 0:
                test_env.render()

            if test_env.GUI:
                sync(i, START, test_env.TIMESTEP)
                
            if done["__all__"]: obs = test_env.reset() # OPTIONAL EPISODE HALT
        
    # logger.save_as_csv("ma") # Optional CSV save
    logger.plot()

    #### Shut down Ray #########################################
    ray.shutdown()