"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` or `VisionAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python fly.py

Notes
-----
The drones move, at different altitudes, along cicular trajectories 
in the X-Y plane, around point (0, -.3).

"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.envs.multi_agent_rl.PayloadCoop import PayloadCoop

import os
import time
import argparse
from datetime import datetime
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
from ray.tune import register_env
from ray.rllib.agents import ppo
from ray.rllib.agents.ppo import PPOTrainer, PPOTFPolicy
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.multi_agent_rl.FlockAviary import FlockAviary
from gym_pybullet_drones.envs.multi_agent_rl.LeaderFollowerAviary import LeaderFollowerAviary
from gym_pybullet_drones.envs.multi_agent_rl.MeetupAviary import MeetupAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync
from experiments.learning import shared_constants


if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=shared_constants.DRONE_MODEL,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=2,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default=shared_constants.PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--vision',             default=False,      type=str2bool,      help='Whether to use VisionAviary (default: False)', metavar='')
    parser.add_argument('--gui',                default=True,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=True,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=False,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate',          default=True,      type=str2bool,      help='Whether to aggregate physics steps (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=True,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=shared_constants.FREQ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=shared_constants.FREQ/shared_constants.AGGR_PHY_STEPS,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=shared_constants.EPISODE_LEN_SEC,          type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    ARGS = parser.parse_args()

    #### Initialize the simulation #############################
    # R = 0.3
    # INIT_XYZS = np.array([[-0.2, 0, 0.5], [0.2, 0, 0.5]])
    # INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin((i/6)*2*np.pi+np.pi/2)-R, 0.5] for i in range(ARGS.num_drones)])
    INIT_XYZS = None
    INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/ARGS.num_drones] for i in range(ARGS.num_drones)])
    AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz/ARGS.control_freq_hz) if ARGS.aggregate else 1
    
    env = PayloadCoop(dest_point=shared_constants.DEST_POINT, drone_model=ARGS.drone,
                         num_drones=ARGS.num_drones,
                         initial_xyzs=INIT_XYZS,
                         initial_rpys=INIT_RPYS,
                         physics=ARGS.physics,
                         neighbourhood_radius=10,
                         freq=ARGS.simulation_freq_hz,
                         aggregate_phy_steps=AGGR_PHY_STEPS,
                         gui=ARGS.gui,
                         episode_len_sec= ARGS.duration_sec)

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=int(ARGS.simulation_freq_hz/AGGR_PHY_STEPS),
                    num_drones=ARGS.num_drones
                    )

    #### Run the simulation ####################################
    env.TRAINING_PHASE = 0
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/ARGS.control_freq_hz))
    action = {i: np.array([0, 1, 0, 1, 1]) for i in range(ARGS.num_drones)}
    START = time.time()
    obs = env.reset()
    return_ = 0
    env.reset()
    env.reset()
    env.reset()
    for i in range(0, int(ARGS.duration_sec*env.SIM_FREQ), AGGR_PHY_STEPS):
        obs, reward, done, info = env.step(action)
        drone0 = env._getDroneStateVector(0)
        drone1 = env._getDroneStateVector(1)
        print(env.step_counter/env.SIM_FREQ, env.EPISODE_LEN_SEC)
        if i%env.SIM_FREQ == 0: # setiap 1 detik
            # print("Jarak Drone 0 dengan drone lain: {}".format((drone1[0:3] - drone0[0:3]).round(3)))
            # print("Hasil Normalisasi:               {}".format((obs[0][5:]).round(3)))
            # print("Jarak Drone 0 dengan goal:       {}".format((env.DEST_POINT - drone0[0:3]).round(3)))
            # print("Hasil Normalisasi:               {}\n".format((obs[0][2:5]).round(3)))
            env.render()
        if ARGS.gui:
            sync(i, START, env.TIMESTEP)
        
        if(done['__all__']):  
            break
        
        
    print(reward, done)
    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()

    # for i in range(env.NUM_DRONES):
    #     temp = ['X', 'Y', 'Z']
    #     t = np.linspace(0, ARGS.duration_sec,ARGS.duration_sec*env.SIM_FREQ)
    #     fig, axs = plt.subplots(3, figsize = (10,5))
    #     plt.suptitle('Drone-{}'.format(i+1))
    #     for j in range(3):
    #         axs[j].plot(t, env.TARGET_HISTORY[i,:,j], label = "target")
    #         axs[j].plot(t, env.POSITION_HISTORY[i,:,j], label = "position")
    #         axs[j].grid()
    #         axs[j].legend()
    #         axs[j].set_xlabel('time (s)')
    #         axs[j].set_ylabel('{} (m)'.format(temp[j]))
    #     plt.show(block = False)
    # plt.show()
 
    #### Plot the simulation results ###########################
    # if ARGS.plot:
    #     logger.plot()
