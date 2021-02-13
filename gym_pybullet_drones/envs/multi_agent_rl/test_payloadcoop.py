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

if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    parser.add_argument('--drone',              default="cf2x",     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=2,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default="pyb",      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--vision',             default=False,      type=str2bool,      help='Whether to use VisionAviary (default: False)', metavar='')
    parser.add_argument('--gui',                default=True,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=True,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=False,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate',          default=False,      type=str2bool,      help='Whether to aggregate physics steps (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=True,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=48,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=300,          type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    ARGS = parser.parse_args()

    #### Initialize the simulation #############################
    R = 0.7
    INIT_XYZS = np.array([[-0.2, 0, 0.5], [0.2, 0, 0.5]])
    # INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin((i/6)*2*np.pi+np.pi/2)-R, 0.5] for i in range(ARGS.num_drones)])
    INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/ARGS.num_drones] for i in range(ARGS.num_drones)])
    AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz/ARGS.control_freq_hz) if ARGS.aggregate else 1
    
    env = PayloadCoop(dest_point=[0, 9, 0.5], drone_model=ARGS.drone,
                         num_drones=ARGS.num_drones,
                         initial_xyzs=INIT_XYZS,
                         initial_rpys=INIT_RPYS,
                         physics=ARGS.physics,
                         neighbourhood_radius=10,
                         freq=ARGS.simulation_freq_hz,
                         aggregate_phy_steps=AGGR_PHY_STEPS,
                         gui=ARGS.gui)

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=int(ARGS.simulation_freq_hz/AGGR_PHY_STEPS),
                    num_drones=ARGS.num_drones
                    )

    #### Run the simulation ####################################
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/ARGS.control_freq_hz))
    action = {str(i): 0 for i in range(ARGS.num_drones)}
    action['0'] =  4
    action['1'] = 4
    START = time.time()

    for i in range(0, int(ARGS.duration_sec*env.SIM_FREQ), AGGR_PHY_STEPS):

        #### Step the simulation ###################################
        obs, reward, done, info = env.step(action)
        print("drone 0: ", obs[0][12:16])
        print("drone 1: ", obs[1][12:16], '\n')
        
        
        ### Log the simulation ####################################
        # for j in range(ARGS.num_drones):
        #     state = env._getDroneStateVector(j)
        #     if action[str(j)] == 0: # Arah X pos
        #         target_pos = state[0:3] + env.K_MOVE * np.array([1, 0, 0])
        #     elif action[str(j)] == 1: # Arah X neg
        #         target_pos = state[0:3] + env.K_MOVE * np.array([-1, 0, 0])
        #     elif action[str(j)] == 2: # Arah Y pos
        #         target_pos = state[0:3] + env.K_MOVE * np.array([0, 1, 0])
        #     elif action[str(j)] == 3: # Arah Y neg
        #         target_pos = state[0:3] + env.K_MOVE * np.array([0, -1, 0])
        #     elif action[str(j)] == 4: # Diam
        #         target_pos = state[0:3]
        #     else:
        #         target_pos = state[0:3]
        #     logger.log(drone=j,
        #                timestamp=i/env.SIM_FREQ,
        #                state= state,
        #                control=np.hstack([target_pos, INIT_RPYS[j, :], np.zeros(6)])
        #                )
        #                #### 12 control targets:                                                                pos_x,
        #                                                                                                      # pos_y,
        #                                                                                                      # pos_z,                                                                                                          
        #                                                                                                      # roll,
        #                                                                                                      # pitch,
        #                                                                                                      # yaw,
        #                                                                                                      # vel_x, 
        #                                                                                                      # vel_y,
        #                                                                                                      # vel_z,
        #                                                                                                      # ang_vel_x,
        #                                                                                                      # ang_vel_y,
        #                                                                                                      # ang_vel_z

        #### Printout ##############################################
        if i%env.SIM_FREQ == 0: #setiap 1 detik
            
            env.render()
            #### Print matrices with the images captured by each drone #
            if ARGS.vision:
                for j in range(ARGS.num_drones):
                    print(obs[str(j)]["rgb"].shape, np.average(obs[str(j)]["rgb"]),
                          obs[str(j)]["dep"].shape, np.average(obs[str(j)]["dep"]),
                          obs[str(j)]["seg"].shape, np.average(obs[str(j)]["seg"])
                          )

        #### Sync the simulation ###################################
        if ARGS.gui:
            sync(i, START, env.TIMESTEP)
        
        if(done['__all__']):  
            break
        

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()

    #### Plot the simulation results ###########################
    if ARGS.plot:
        logger.plot()
