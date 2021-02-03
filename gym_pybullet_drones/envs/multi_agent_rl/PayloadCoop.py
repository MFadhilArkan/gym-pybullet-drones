import os
from datetime import datetime
import numpy as np
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.utils import nnlsRPM
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl

class PayloadCoop(BaseAviary, MultiAgentEnv):
    
    ################################################################################

    def __init__(self,
                 dest_point = None,
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
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a generic multi-agent RL environment.

        Attributes `vision_attributes` and `dynamics_attributes` are selected
        based on the choice of `obs` and `act`; `obstacles` is set to True 
        and overridden with landmarks for vision applications; 
        `user_debug_gui` is set to False for performance.

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
            The type of action space (1 or 3D; RPMS, thurst and torques, waypoint or velocity with PID control; etc.)

        """
        if num_drones != 2:
            print("Jumlah drone tidak 2")
            exit()
        if dest_point == None:
            print("Destination point belum ada")
            exit()
        self.dest_point = dest_point
        vision_attributes = True if obs == ObservationType.RGB else False
        dynamics_attributes = True if act in [ActionType.DYN, ActionType.ONE_D_DYN] else False
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        self.EPISODE_LEN_SEC = 5
        #### Create integrated controllers #########################
        if act in [ActionType.PID, ActionType.VEL, ActionType.ONE_D_PID]:
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
            if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
                self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(num_drones)]
            elif drone_model == DroneModel.HB:
                self.ctrl = [SimplePIDControl(drone_model=DroneModel.HB) for i in range(num_drones)]
            else:
                print("[ERROR] in BaseMultiagentAviary.__init()__, no controller is available for the specified drone_model")
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
                         obstacles=True, # Add obstacles for RGB observations and/or FlyThruGate
                         user_debug_gui=False, # Remove of RPM sliders from all single agent learning aviaries
                         vision_attributes=vision_attributes,
                         dynamics_attributes=dynamics_attributes
                         )
        #### Set a limit on the maximum target speed ###############
        if act == ActionType.VEL:
            self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000/3600)

    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.

        Only if the observation is of type RGB, 4 landmarks are added.
        Overrides BaseAviary's method.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            p.loadURDF("block.urdf",
                       [1, 0, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
            p.loadURDF("cube_small.urdf",
                       [0, 1, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
            p.loadURDF("duck_vhacd.urdf",
                       [-1, 0, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
            p.loadURDF("teddy_vhacd.urdf",
                       [0, -1, .1],
                       p.getQuaternionFromEuler([0, 0, 0]),
                       physicsClientId=self.CLIENT
                       )
        else:
            pass

    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        dict[int, ndarray]
            A Dict() of Box() of size 1, 3, or 3, depending on the action type,
            indexed by drone Id in integer format.

        """
        # if self.ACT_TYPE in [ActionType.RPM, ActionType.DYN, ActionType.VEL]:
        #     size = 4
        # elif self.ACT_TYPE==ActionType.PID:
        #     size = 3
        # elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_DYN, ActionType.ONE_D_PID]:
        #     size = 1
        # else:
        #     print("[ERROR] in BaseMultiagentAviary._actionSpace()")
        #     exit()
        # return spaces.Dict({i: spaces.Box(low=-1*np.ones(size),
        #                                   high=np.ones(size),
        #                                   dtype=np.float32
        #                                   ) for i in range(self.NUM_DRONES)})
        return spaces.Dict({i: spaces.Discrete(4) for i in range(self.NUM_DRONES)})

    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Parameter `action` is processed differenly for each of the different
        action types: the input to n-th drone, `action[n]` can be of length
        1, 3, or 4, and represent RPMs, desired thrust and torques, or the next
        target position to reach using PID control.

        Parameter `action` is processed differenly for each of the different
        action types: `action` can be of length 1, 3, or 4 and represent 
        RPMs, desired thrust and torques, the next target position to reach 
        using PID control, a desired velocity vector, etc.

        Parameters
        ----------
        action : dict[str, ndarray]
            The input action for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        K_MOVE = 0.001
        rpm = np.zeros((self.NUM_DRONES,4))
        for k, v in action.items():
            if self.ACT_TYPE == ActionType.JOYSTICK:
                state = self._getDroneStateVector(int(k))
                if v == 1: # Arah X pos
                    target_pos = state[0:3] + K_MOVE * [1, 0, 0]
                elif v == 2: # Arah X neg
                    target_pos = state[0:3] + K_MOVE * [-1, 0, 0]
                elif v == 3: # Arah Y pos
                    target_pos = state[0:3] + K_MOVE * [0, 1, 0]
                elif v == 4: # Arah Y neg
                    target_pos = state[0:3] + K_MOVE * [0, -1, 0]
                else:
                    target_pos = state[0:3]
                    print("Aksi tidak diketahui, drone ke-{} akan diam\n".format(k))
                rpm_k, _, _ = self.ctrl[int(k)].computeControl(control_timestep=self.AGGR_PHY_STEPS*self.TIMESTEP, 
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=target_pos
                                                        )
                rpm[int(k),:] = rpm_k
            else:
                print("[ERROR] in BaseMultiagentAviary._preprocessAction(), ACT_TYPE is not JOYSTICK")
                exit()
        return rpm

    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        dict[int, ndarray]
            A Dict with NUM_DRONES entries indexed by Id in integer format,
            each a Box() os shape (H,W,4) or (12,) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.KIN:
            # 12 + 4 state deteksi rintangan, 3 state jarak dengan drone lain, 3 state jarak dengan tujuan, 
            return spaces.Dict({i: spaces.Box(low=np.array([-1,-1,0, -1,-1,-1, -1,-1,-1, -1,-1,-1, 0,0,0,0, 0,0,0, 0,0,0]),
                                              high=np.array([1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1,1, 1,1,1, 1,1,1]),
                                              dtype=np.float32
                                              ) for i in range(self.NUM_DRONES)})
        else:
            print("[ERROR] in BaseMultiagentAviary._observationSpace(), OBS_TYPE is not KIN_PAY")
    
    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        dict[int, ndarray]
            A Dict with NUM_DRONES entries indexed by Id in integer format,
            each a Box() os shape (H,W,4) or (12,) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.KIN: 
            #### 12 + 4 state deteksi rintangan, 3 state jarak dengan drone lain, 3 state jarak dengan tujuan, 
            obs_22 = np.zeros((self.NUM_DRONES,22))
            
            for i in range(self.NUM_DRONES):
                obs_kin = self._getDroneStateVector(i) # Kinematic drone 
                obs_obstacle = _isObstacleNear(position = obs_kin[0:3]) # Obstacle
                obs_dist_drone = self._getDroneStateVector((i + 1) % 2)[0:3] - obs_kin[0:3] # Jarak dengan drone lain
                obs_dist_dest = self.dest_point - obs_kin[0:3] # Jarak dengan posisi tujuan
                obs_22[i, :] = np.hstack([obs_kin[0:3], obs_kin[7:10], obs_kin[10:13], obs_kin[13:16], obs_obstacle, obs_dist_drone, obs_dist_dest]).reshape(22,)
                
            return {i: obs_22[i, :] for i in range(self.NUM_DRONES)}
            ############################################################
        else:
            print("[ERROR] in BaseMultiagentAviary._computeObs()")

    ################################################################################

    def _isObstacleNear(self, position):
        raise NotImplementedError

    ################################################################################

    def _clipAndNormalizeState(self, state):
        """Normalizes a drone's state to the [-1,1] range.

        Must be implemented in a subclass.

        Parameters
        ----------
        state : ndarray
            Array containing the non-normalized state of a single drone.

        """
        raise NotImplementedError
    
    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Returns
        -------
        dict[int, float]
            The reward value for each drone.

        """
        raise NotImplementedError

    ################################################################################
    
    def _computeDone(self):
        raise NotImplementedError

    ################################################################################
    
    def _computeInfo(self):
        raise NotImplementedError
    
    ################################################################################

    def _clipAndNormalizeState(self,state):
        raise NotImplementedError
    
    ################################################################################
    
    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_xy,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.
        
        """
        raise NotImplementedError
