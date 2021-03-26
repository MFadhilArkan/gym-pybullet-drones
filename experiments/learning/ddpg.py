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
import ray
from ray import tune
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune import register_env
from ray.rllib.agents import ppo, ddpg
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
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.multi_agent_rl.PayloadCoop import PayloadCoop
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.Logger import Logger
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer
import shared_constants
import tensorflow as tf
import tensorflow.keras.layers as layers
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC

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


current_phase=0
class FillInActions(DefaultCallbacks):
    # def on_postprocess_trajectory(self, worker, episode, agent_id, policy_id, policies, postprocessed_batch, original_batches, **kwargs):
    #     to_update = postprocessed_batch[SampleBatch.CUR_OBS]
    #     other_id = 1 if agent_id == 0 else 0
    #     action_encoder = ModelCatalog.get_preprocessor_for_space( 
    #                                                              Box(-np.inf, np.inf, (ACTION_VEC_SIZE,), np.float32) # Unbounded
    #                                                              )

    #     _, opponent_batch = original_batches[other_id]
    #     opponent_actions = np.array([action_encoder.transform(a) for a in opponent_batch[SampleBatch.ACTIONS]])
    #     to_update[:, -ACTION_VEC_SIZE:] = opponent_actions

    def on_train_result(self, trainer, result):
        global current_phase
        print("Manage Curriculum callback called on phase {}".format(current_phase))
        if result["episode_reward_mean"] > shared_constants.RWD_ARRIVE / 1.5:
            current_phase+=1
            trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.set_phase(current_phase)))


def central_critic_observer(agent_obs, **kw):
    new_obs = {
        0: {
            "own_obs": agent_obs[0],
            "opponent_obs": agent_obs[1],
            "opponent_action": np.zeros(ACTION_VEC_SIZE),
        },
        1: {
            "own_obs": agent_obs[1],
            "opponent_obs": agent_obs[0],
            "opponent_action": np.zeros(ACTION_VEC_SIZE),
        },
    }
    return new_obs

if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Multi-agent reinforcement learning experiments script')
    parser.add_argument('--num_drones',  default=shared_constants.NUM_DRONES,            type=int,                                                                 help='Number of drones (default: 2)', metavar='')
    parser.add_argument('--env',         default=shared_constants.ENV,      type=str,             choices=['leaderfollower', 'flock', 'meetup', 'payloadcoop'],      help='Help (default: ..)', metavar='')
    parser.add_argument('--obs',         default=shared_constants.OBS,        type=ObservationType,                                                     help='Help (default: ..)', metavar='')
    parser.add_argument('--act',         default=shared_constants.ACT,  type=ActionType,                                                          help='Help (default: ..)', metavar='')
    parser.add_argument('--algo',        default='cc',         type=str,             choices=['cc'],                                     help='Help (default: ..)', metavar='')
    parser.add_argument('--workers',     default=7,            type=int,                                                                 help='Help (default: ..)', metavar='')        
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
    observer_space = Dict({
        "own_obs": temp_env.observation_space[0],
        "opponent_obs": temp_env.observation_space[0],
        "opponent_action": temp_env.action_space[0],
    })
    action_space = temp_env.action_space[0]
    OWN_OBS_VEC_SIZE = observer_space["own_obs"].shape[0]
    ACTION_VEC_SIZE = action_space.shape[0]
    ModelCatalog.register_custom_model("cc_model",CustomTorchCentralizedCriticModel)
    config = {
        "env": temp_env_name,
        "num_workers": 0 + ARGS.workers,
        "num_envs_per_worker": 4,
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
    stop = {
        "episodes_total": 100000, # 8000,
        # "episode_reward_mean": 0,
        # "training_iteration": 0,
    }

    ##################### TRAINING ###############################
    with open(filename+'/training_constant.txt', 'w+') as f:
        for attr in dir(shared_constants):
            value = getattr(shared_constants, attr)
            if(not callable(getattr(shared_constants, attr)) and not attr.startswith("__")):
                f.write("{} : {}\n".format(attr, value))

    trainer = ddpg.DDPGTrainer
    results = tune.run(
        trainer,
        stop=stop,
        config=config,
        verbose=3,
        checkpoint_at_end=True,
        local_dir=filename,
        checkpoint_freq=10,
        # max_failures=-1,
        # restore=open("experiments/learning/results/save-payloadcoop-2-cc-kin-xyz_yaw-03.11.2021_20.57.58/checkpoint.txt").read()
        # restore = "/home/mahendra/git/gym-pybullet-drones/experiments/learning/results/save-payloadcoop-2-cc-payload_one_sensor-vel_yaw-03.12.2021_06.06.40/DDPG/DDPG_this-aviary-v0_71fdc_00000_0_2021-03-12_06-06-43/checkpoint_10/checkpoint-10"
    )
    # check_learning_achieved(results, 1.0)
    checkpoints = results.get_trial_checkpoints_paths(trial=results.get_best_trial('episode_reward_mean',
                                                                                   mode='max'
                                                                                   ),
                                                      metric='episode_reward_mean'
                                                      )
    with open(filename+'/checkpoint.txt', 'w+') as f:
        f.write(checkpoints[0][0])
    ###########################################################
    
    ray.shutdown()