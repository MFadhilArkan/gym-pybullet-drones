import math
import numpy as np
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.envs.multi_agent_rl.BaseMultiagentAviary import BaseMultiagentAviary

class LeaderFollowerAviary(BaseMultiagentAviary):
    """Multi-agent RL problem: leader-follower."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=2,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False, 
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM):
        """Initialization of a multi-agent RL environment.

        Using the generic multi-agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record, 
                         obs=obs,
                         act=act
                         )

    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value(s).

        Returns
        -------
        dict[int, float]
            The reward value for each drone.

        """
        rewards = {}
        drone_ids = self.getDroneIds()
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        rewards[0] = -10 * np.linalg.norm(self.DEST_POINT[0] - states[0, 0])
        rewards[0] += -10 * np.linalg.norm(self.DEST_POINT[1] - states[0, 1])
        rewards[0] += -10 * np.linalg.norm(self.DEST_POINT[2] - states[0, 2])
        #print(states[:,0:3])
        #if(self._isHitEverything(0)):
        #      reward[0] += -100
        # rewards[1] = -1 * np.linalg.norm(np.array([states[1, 0], states[1, 1], 0.5]) - states[1, 0:3])**2 # DEBUG WITH INDEPENDENT REWARD 
        for i in range(1, self.NUM_DRONES):
            rewards[i] = -10 * np.linalg.norm(self.DEST_POINT[0] + self.INIT_XYZS[i,0]- self.INIT_XYZS[0,0] - states[i, 0])
            rewards[i] += -10 * np.linalg.norm(self.DEST_POINT[1] + self.INIT_XYZS[i,1]- self.INIT_XYZS[0,1] - states[i, 1])
            rewards[i] += -10 * np.linalg.norm(self.INIT_XYZS[0,2] - states[i, 2])
              #print(states[:,0:3])
            #if(self._isHitEverything(drone_ids)):
            #  rewards[i] += -100
            #  print(states[:,0:3])
        if(self._isArrive(drone_ids)):
          print("position:",states[:,0:3])
          print("velocity:",states[:,6:9])
          rewards[0] += 1000 - 250*(states[0,6]+states[0,7])
          rewards[1] += 1000 - 250*(states[1,6]+states[1,7])
        return rewards

    ################################################################################
    
    def _computeDone(self):
        drone_ids = self.getDroneIds()
        bool_val = self._isDroneTooFar(drone_ids) #\
            #or self._isHitEverything(drone_ids)
        bool_val = bool_val or (self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC)
        done = {i: bool_val for i in range(self.NUM_DRONES)}
        done["__all__"] = True if True in done.values() else False
        return done

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[int, dict[]]
            Dictionary of empty dictionaries.

        """
        return {i: {} for i in range(self.NUM_DRONES)}

    ################################################################################
