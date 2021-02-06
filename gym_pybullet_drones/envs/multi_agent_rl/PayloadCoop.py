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
                 act: ActionType=ActionType.JOYSTICK
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

        self.MAX_DISTANCE_BETWEEN_DRONE = 2
        self.dest_point = dest_point
        self.OBSTACLE_IDS = []
        vision_attributes = True if obs == ObservationType.RGB else False
        dynamics_attributes = True if act in [ActionType.DYN, ActionType.ONE_D_DYN] else False
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        self.EPISODE_LEN_SEC = 10
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

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        dict[int, ndarray]
            A Dict with NUM_DRONES entries indexed by Id in integer format,
            each a Box() os shape (H,W,4) or (12,) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.KIN:
            # 12 + 4 state deteksi rintangan [x+, x-, y+, y-], 3 state jarak dengan drone lain, 3 state jarak dengan tujuan, 
            return spaces.Dict({i: spaces.Box(low=np.array([-1,-1,0, -1,-1,-1, -1,-1,-1, -1,-1,-1, 0,0,0,0, 0,0,0, 0,0,0]),
                                              high=np.array([1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1,1, 1,1,1, 1,1,1]),
                                              dtype=np.float32
                                              ) for i in range(self.NUM_DRONES)})
        else:
            print("[ERROR] in BaseMultiagentAviary._observationSpace(), OBS_TYPE is not KIN")

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
                obs_obstacle = self._isObstacleNear(i) # Obstacle
                obs_dist_drone = self._getDroneStateVector((i + 1) % 2)[0:3] - obs_kin[0:3] # Jarak dengan drone lain
                obs_dist_dest = self.dest_point - obs_kin[0:3] # Jarak dengan posisi tujuan
                obs_22[i, :] = np.hstack([obs_kin[0:3], obs_kin[7:10], obs_kin[10:13], obs_kin[13:16], obs_obstacle, obs_dist_drone, obs_dist_dest]).reshape(22,)

            obs = self._clipAndNormalizeState(obs)   
            return {i: obs_22[i, :] for i in range(self.NUM_DRONES)}
            ############################################################
        else:
            print("[ERROR] in PayloadCoop._computeObs()")

    def _computeReward(self):
        """Computes the current reward value(s).

        Returns
        -------
        dict[int, float]
            The reward value for each drone.

        """
        drone_ids = self.getDroneIds()
        reward = 0
        rwd_hit = -100
        rwd_toofar_drone = -100
        rwd_arrive = 100
        rwd_time = -1
        rwd_rpm = 0.001

        # # Menabrak obstacle
        # if(_isHitObstacle(drone_ids)):
        #     reward += rwd_hit_obs

        # # Menabrak drone lain
        # if(_isHitDrone(drone_ids)):
        #     reward += rwd_hit_drone

        # Menabrak sesuatu
        if(self._isHitEverything(drone_ids)):
            reward += rwd_hit

        # Kedua drone berjauhan
        if(self._isDroneTooFar(drone_ids)):
            reward += rwd_toofar_drone   

        # Mencapai tujuan
        if(self._isArrive(drone_ids)):
            reward += rwd_arrive

        # Waktu tempuh
        reward += rwd_time

        # Jumlah aksi kumulatif
        RPM_eq = 16073 #sqrt(mg/4Ct)
        reward += rwd_rpm * np.sum(drone_states[16:20] - RPM_eq)

        return reward

    ################################################################################
    
    def _computeDone(self):
        drone_ids = self.getDroneIds()
        bool_val = False
        for i in drone_ids:
            bool_val = self._isArrive(i) \
            or self._isHitEverything(i) \
            or self._isDroneTooFar(i) \ 
        self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC


        bool_val = True if self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC else False
        done = {i: bool_val for i in range(self.NUM_DRONES)}
        done["__all__"] = True if True in done.values() else False

    ################################################################################
    
    def _computeInfo(self):
        return {i: {} for i in range(self.NUM_DRONES)}

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

     def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (22,)- 12 + 4 state deteksi rintangan, 3 state jarak dengan drone lain, 3 state jarak dengan tujuan, 

        Returns
        -------
        ndarray
            (22,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_LIN_VEL_XY = 3 
        MAX_LIN_VEL_Z = 1
        MAX_XY = 10
        MAX_Z = 5
        MAX_DIST_GOAL = np.sqrt(2*MAX_XY**2)

        # MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        # MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[3:5], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[6:8], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[8], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_xy,
                                               clipped_pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z
                                               )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[5] / np.pi # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[9:12]/np.linalg.norm(state[9:12]) if np.linalg.norm(state[9:12]) != 0 else state[9:12]

        normalized_dist_betw_drone = state[16:19] / self.MAX_DISTANCE_BETWEEN_DRONE
        normalized_dist_goal = state[19:] / self.MAX_DIST_GOAL

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[12:16],
                                      normalized_dist_betw_drone,
                                      normalize_dist_goal
                                      ]).reshape(22,)

        return norm_and_clipped
    
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
        if not(clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not(clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not(clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))
    
    ################################################################################

    def _isHitEverything(self, drone_ids):
        for i in range(len(drone_ids)):
            a, b = p.getAABB(drone_ids[i]) # Melihat batas posisi collision drone ke i
            list_obj = p.getOverlappingObjects(a, b) # Melihat objek2 yang ada di batas posisi collision
            if(len(list_obj) >= 6) # 1 Quadcopter memiliki 6 link/bagian
                return True
        return False

    ################################################################################

    def _isDroneTooFar(self, drone_ids, max_dist = self.MAX_DISTANCE_BETWEEN_DRONE):
        # Looping untuk setiap pair of drone
        for i in range(len(drone_ids)):
            for j in range(i + 1, len(drone_ids)):
                dr1_states = self._getDroneStateVector(drone_ids[i])
                dr2_states = self._getDroneStateVector(drone_ids[j])
                dist = np.linalg.norm(dr1_states[0:3] - dr2_states[0:3])
                if (dist > max_dist_drone):
                    return True
        return False

    def _isArrive(self, drone_ids, tol = 0.01, dest = self.dest_point):
        # Menghitung centroid of points dari kumpulan drone
        centroid = [0, 0, 0]
        for i in range(len(drone_ids)):
            dr_states = self._getDroneStateVector(drone_ids[i])
            centroid += dr_states[0:3]
        centroid /= len(drone_ids)
        if(np.linalg.norm(centroid - dest) < tol):
            return True
        else:
            return False
    
    ################################################################################

    def _addObstacles(self, position):
        """Add obstacles to the environment.

        Only if the observation is of type RGB, 4 landmarks are added.
        Overrides BaseAviary's method.

        """
        id_ = p.loadURDF("cube_small.urdf",
                    position,
                    p.getQuaternionFromEuler([0, 0, 0]),
                    physicsClientId=self.CLIENT
                    )
        self.OBSTACLE_IDS.append(id_)    

    ################################################################################

     def _isObstacleNear(self, drone_id, sensor_dist = 2, min_dist_parallel = 0.5):
        obstacle_state = np.zeros(4)
        drone_state = self._getDroneStateVector(drone_id)
        for obst_id in self.OBSTACLE_IDS:
            list_cp = p.getClosestPoints(drone_id, obstacle_state, sensor_dist)
            if(len(list_cp) != 0): # ada obstacle didekat drone
                x_dr, y_dr, z_dr = drone_state[0:3]
                x_ob, y_ob, z_ob = list_cp[0][6]
                if(np.abs(y_ob - y_dr) < min_dist_parallel): # obstacle dan drone sejajar sumbu y
                    if(x_ob >= x_dr):
                        obstacle_state[0] = 1
                    else:
                        obstacle_state[1] = 1

                elif(np.abs(x_ob - x_dr) < min_dist_parallel):
                    if(y_ob >= y_dr):
                        obstacle_state[2] = 1
                    else:
                        obstacle_state[3] = 1
        return obstacle_state