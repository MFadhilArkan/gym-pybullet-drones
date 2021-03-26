import os
from datetime import datetime
import numpy as np
from gym import spaces
import pybullet as p
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.utils import nnlsRPM
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl

class BaseMultiagentAviary(BaseAviary, MultiAgentEnv):
    """Base multi-agent environment class for reinforcement learning."""
    
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
        if num_drones < 2:
            print("[ERROR] in BaseMultiagentAviary.__init__(), num_drones should be >= 2")
            exit()
        if act == ActionType.TUN:
            print("[ERROR] in BaseMultiagentAviary.__init__(), ActionType.TUN can only used with BaseSingleAgentAviary")
            exit()
        vision_attributes = True if obs == ObservationType.RGB else False
        dynamics_attributes = True if act in [ActionType.DYN, ActionType.ONE_D_DYN] else False
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        self.EPISODE_LEN_SEC = 5
        self.K_MOVE = 0.075
        #### Create integrated controllers #########################
        if act in [ActionType.PID, ActionType.VEL, ActionType.ONE_D_PID, ActionType.JOYSTICK, ActionType.XYZ_YAW, ActionType.XY_YAW]:
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
            if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
                self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(num_drones)]
            elif drone_model in [DroneModel.HB, DroneModel.ARDRONE2]:
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


        self.MAX_DISTANCE_BETWEEN_DRONE = 0.5
        self.MAX_XY = 5
        self.MAX_Z = 1
        self.DEST_POINT = np.array([0,1,self.INIT_XYZS[0,2]])
        self.Z_CONST = self.DEST_POINT[2]
        self.OBSTACLE_IDS = []

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
        if(self.ACT_TYPE == ActionType.JOYSTICK):
            return spaces.Dict({i: spaces.Discrete(5) for i in range(self.NUM_DRONES)})

        if self.ACT_TYPE in [ActionType.RPM, ActionType.DYN, ActionType.VEL]:
            size = 4
        elif self.ACT_TYPE==ActionType.PID:
            size = 2
        elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_DYN, ActionType.ONE_D_PID]:
            size = 1
        else:
            print("[ERROR] in BaseMultiagentAviary._actionSpace()")
            exit()
        return spaces.Dict({i: spaces.Box(low=-1*np.ones(size),
                                          high=np.ones(size),
                                          dtype=np.float32
                                          ) for i in range(self.NUM_DRONES)})

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
        rpm = np.zeros((self.NUM_DRONES,4))
        v_act = np.zeros((2))
        #print(action)
        for k, v in action.items():
            if self.ACT_TYPE == ActionType.RPM: 
                rpm[int(k),:] = np.array(self.HOVER_RPM * (1+0.05*v))
            elif self.ACT_TYPE == ActionType.DYN: 
                rpm[int(k),:] = nnlsRPM(thrust=(self.GRAVITY*(v[0]+1)),
                                        x_torque=(0.05*self.MAX_XY_TORQUE*v[1]),
                                        y_torque=(0.05*self.MAX_XY_TORQUE*v[2]),
                                        z_torque=(0.05*self.MAX_Z_TORQUE*v[3]),
                                        counter=self.step_counter,
                                        max_thrust=self.MAX_THRUST,
                                        max_xy_torque=self.MAX_XY_TORQUE,
                                        max_z_torque=self.MAX_Z_TORQUE,
                                        a=self.A,
                                        inv_a=self.INV_A,
                                        b_coeff=self.B_COEFF,
                                        gui=self.GUI
                                        )
            elif self.ACT_TYPE == ActionType.JOYSTICK: 
                state = self._getDroneStateVector(int(k))
                if v == 0: # Arah X pos
                    v_act = self.K_MOVE * np.array([1, 0])
                elif v == 1: # Arah Y pos
                    v_act = self.K_MOVE * np.array([0, 1])
                elif v == 2: # Arah X neg
                    v_act = self.K_MOVE * np.array([-1, 0])
                elif v == 3: # Arah Y neg
                    v_act = self.K_MOVE * np.array([0, -1])
                elif v == 4: # Diam
                    v_act = self.K_MOVE * np.array([0, 0])
                else:
                    v_act = self.K_MOVE * np.array([0, 0])
                    print("Aksi tidak diketahui, drone ke-{} akan diam\n".format(k))
                rpm_k, _, _ = self.ctrl[int(k)].computeControl(control_timestep=self.AGGR_PHY_STEPS*self.TIMESTEP, 
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=np.hstack([state[0:2]+v_act[0:2],self.INIT_XYZS[int(k),2]]),
                                                        target_rpy=np.array([0,0,state[9]])
                                                        )
                rpm[int(k),:] = rpm_k
            elif self.ACT_TYPE == ActionType.VEL:
                state = self._getDroneStateVector(int(k))
                if np.linalg.norm(v[0:3]) != 0:
                    v_unit_vector = v[0:3] / np.linalg.norm(v[0:3])
                else:
                    v_unit_vector = np.zeros(3)
                temp, _, _ = self.ctrl[int(k)].computeControl(control_timestep=self.AGGR_PHY_STEPS*self.TIMESTEP, 
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=state[0:3], # same as the current position
                                                        target_rpy=np.array([0,0,state[9]]), # keep current yaw
                                                        target_vel=self.SPEED_LIMIT * np.abs(v[3]) * v_unit_vector # target the desired velocity vector
                                                        )
                rpm[int(k),:] = temp
            elif self.ACT_TYPE == ActionType.ONE_D_RPM: 
                rpm[int(k),:] = np.repeat(self.HOVER_RPM * (1+0.05*v), 4)
            elif self.ACT_TYPE == ActionType.ONE_D_DYN: 
                rpm[int(k),:] = nnlsRPM(thrust=(self.GRAVITY*(1+0.05*v[0])),
                                        x_torque=0,
                                        y_torque=0,
                                        z_torque=0,
                                        counter=self.step_counter,
                                        max_thrust=self.MAX_THRUST,
                                        max_xy_torque=self.MAX_XY_TORQUE,
                                        max_z_torque=self.MAX_Z_TORQUE,
                                        a=self.A,
                                        inv_a=self.INV_A,
                                        b_coeff=self.B_COEFF,
                                        gui=self.GUI
                                        )
            elif self.ACT_TYPE == ActionType.ONE_D_PID:
                state = self._getDroneStateVector(int(k))
                rpm, _, _ = self.ctrl[k].computeControl(control_timestep=self.AGGR_PHY_STEPS*self.TIMESTEP, 
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=state[0:3]+0.1*np.array([0,0,v[0]])
                                                        )
                rpm[int(k),:] = rpm

            elif self.ACT_TYPE == ActionType.PID: 
                state = self._getDroneStateVector(int(k))
                rpm_k, _, _ = self.ctrl[int(k)].computeControl(control_timestep=self.AGGR_PHY_STEPS*self.TIMESTEP, 
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=np.hstack([state[0:2]+0.1*v[0:2],self.INIT_XYZS[0,2]]),
                                                        target_rpy=np.hstack([0, 0, state[9]])
                                                        )
                rpm[int(k),:] = rpm_k 

            else:
                print("[ERROR] in BaseMultiagentAviary._preprocessAction()")
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
        if self.OBS_TYPE in [ObservationType.KIN, ObservationType.PAYLOAD, ObservationType.PAYLOAD_Z_CONST, ObservationType.NOOBS]:
            if self.OBS_TYPE == ObservationType.KIN:
                #(x,y,z, r,p,y, x_dot, y_dot, z_dot, r_dot,p_dot,y_dot, ob_x+,ob_y+,ob_x-,ob_y-, x_dest-x,y_dest-y,z_dest-z,  x_dr2-x,y_dr2-y,z_dr2-z .....)
                #12 + 4 state obstacle, 3 state distance to dest_point, 3N state distance between drone,
                dist_drone = np.ones((3*(self.NUM_DRONES - 1)))
                low = np.array([-1,-1,0, -1,-1,-1, -1,-1,-1, -1,-1,-1, 0,0,0,0, -1,-1,-1])
                high = np.array([1,1,1,   1,1,1,    1,1,1,    1,1,1,   1,1,1,1,  1,1,1])

            elif self.OBS_TYPE == ObservationType.PAYLOAD_Z_CONST:
                #(ob_x+,ob_y+,ob_x-,ob_y-, x_dest-x,y_dest-y, x_dr2-x,y_dr2-y .....)
                #4 state obstacle, 2 state distance to dest_point, 2N state distance between drone,
                dist_drone = np.ones((2*(self.NUM_DRONES - 1)))
                low = np.array([0,0,0,0, -1,-1])
                high = np.array([1,1,1,1,  1,1])

            elif self.OBS_TYPE == ObservationType.PAYLOAD:
                #(ob_x+,ob_y+,ob_x-,ob_y-, x_dest-x,y_dest-y, x_dr2-x,y_dr2-y .....)
                #4 state obstacle, 3 state distance to dest_point, 3N state distance between drone,
                dist_drone = np.ones((3*(self.NUM_DRONES - 1)))
                low = np.array([0,0,0,0, -1,-1,-1])
                high = np.array([1,1,1,1,  1,1,1])

            elif self.OBS_TYPE == ObservationType.NOOBS:
                #(ob_x+,ob_y+,ob_x-,ob_y-, x_dest-x,y_dest-y, x_dr2-x,y_dr2-y .....)
                #4 state obstacle, 3 state distance to dest_point, 3N state distance between drone,
                dist_drone = np.ones((3*(self.NUM_DRONES - 1)))
                low = np.array([-1,-1,-1])
                high = np.array([1,1,1])    

            low = np.hstack([low, -1 * dist_drone])
            high = np.hstack([high, dist_drone])
            return spaces.Dict({i: spaces.Box(low=low,
                                                high=high,
                                                dtype=np.float32
                                                ) for i in range(self.NUM_DRONES)})
        else:
            print("[ERROR] in PayloadCoop._observationSpace()")
    
    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        dict[int, ndarray]
            A Dict with NUM_DRONES entries indexed by Id in integer format,
            each a Box() os shape (H,W,4) or (12,) depending on the observation type.

        """
        if self.OBS_TYPE in [ObservationType.KIN, ObservationType.PAYLOAD, ObservationType.PAYLOAD_Z_CONST, ObservationType.NOOBS]:
            obs_all = np.zeros((self.NUM_DRONES, 19+3*(self.NUM_DRONES - 1)), dtype = np.float32)
            for i in range(self.NUM_DRONES):
                state = self._getDroneStateVector(i)
                obs_all[i, 0:3] = state[0:3] # pos
                obs_all[i, 3:6] = state[7:10] # rpy
                obs_all[i, 6:9] = state[10:13] # pos_dot
                obs_all[i, 9:12] = state[13:16] # rpy_dot
                obs_all[i, 12:16] = self._isObstacleNear(i) # obstacle sensor
                obs_all[i, 16:19] = self.DEST_POINT - state[0:3] # distance to DEST_POINT
                obs_all[i, 19:] = self._getDistBetweenAllDrone(i) # distance between drone
                obs_all[i, :] = self._clipAndNormalizeState(obs_all[i, :])             
        
            if self.OBS_TYPE == ObservationType.KIN:
                return {i: obs_all[i, :] for i in range(self.NUM_DRONES)}

            elif self.OBS_TYPE == ObservationType.PAYLOAD_Z_CONST:
                mask = np.arange(3*(self.NUM_DRONES - 1))
                mask = (mask+1) % 3 != 0 # remove dist_betw_drone z state index
                mask = np.hstack([[False]*12, [True]*6, False, mask]) # 12 drone state, 4 obst + 3 dist2dest                
                return {i: obs_all[i, mask] for i in range(self.NUM_DRONES)}

            elif self.OBS_TYPE == ObservationType.PAYLOAD:
                return {i: obs_all[i, 12:] for i in range(self.NUM_DRONES)}

            elif self.OBS_TYPE == ObservationType.NOOBS:
                return {i: obs_all[i, 16:] for i in range(self.NUM_DRONES)}
        else:
            print("[ERROR] in PayloadCoop._computeObs()")

    ################################################################################

    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Must be implemented in a subclass.

        Parameters
        ----------
        state : ndarray
            Array containing the non-normalized state of a single drone.

        """
        COEF_TOL = 1.3
        MAX_LIN_VEL_XY = self.MAX_SPEED_KMH /3.6 * COEF_TOL
        MAX_LIN_VEL_Z = self.MAX_SPEED_KMH /3.6 * COEF_TOL
        MAX_ANG_VEL = self.MAX_SPEED_KMH /3.6 * COEF_TOL * 0.9 # dikali jari jari
        MAX_XY = self.MAX_XY * COEF_TOL
        MAX_Z = self.MAX_Z * COEF_TOL
        MAX_DIST_GOAL = np.sqrt(2*MAX_XY**2) * COEF_TOL
        MAX_DISTANCE_BETWEEN_DRONE = self.MAX_DISTANCE_BETWEEN_DRONE * COEF_TOL

        # MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        # MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[3:5], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[6:8], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[8], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)
        clipped_ang_vel = np.clip(state[9:12], -MAX_ANG_VEL, MAX_ANG_VEL)
        clipped_dist_goal = np.clip(state[16:19], -MAX_DIST_GOAL, MAX_DIST_GOAL)
        clipped_dist_drone = np.clip(state[19:], -MAX_DISTANCE_BETWEEN_DRONE, MAX_DISTANCE_BETWEEN_DRONE)
        

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_xy,
                                               clipped_pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z
                                               )
        state[0:2] = clipped_pos_xy / MAX_XY
        state[2] = clipped_pos_z / MAX_Z
        state[3:5] = clipped_rp / MAX_PITCH_ROLL
        state[5] = state[5] / np.pi # No reason to clip
        state[6:8] = clipped_vel_xy / MAX_LIN_VEL_XY
        state[8] = clipped_vel_z / MAX_LIN_VEL_XY
        state[9:12] = clipped_ang_vel / MAX_ANG_VEL
        state[16:19] = clipped_dist_goal / MAX_DIST_GOAL
        state[19:]= clipped_dist_drone / MAX_DISTANCE_BETWEEN_DRONE                                       
        return state

    def _getDistBetweenAllDrone(self, drone_id):
        obst_dist = np.zeros((3*(self.NUM_DRONES - 1)), dtype = np.float32)
        state_drone_main = self._getDroneStateVector(drone_id)
        j = 0
        for i in range(self.NUM_DRONES):
            if(i != drone_id):
                state_drone_sec = self._getDroneStateVector(i)
                obst_dist[3*j: 3*j+3] = state_drone_sec[0:3] - state_drone_main[0:3]
                j += 1
        return obst_dist       

    def _isObstacleNear(self, drone_id, max_sensor_dist = 2, max_sensor_angle = 10): # drone_id (ordinal)
        if(self.ACT_TYPE in [ActionType.JOYSTICK]):
            obstacle_state = np.zeros(4)
            drone_state = self._getDroneStateVector(drone_id)
            for obst_id in list(self.DRONE_IDS[:drone_id]) + list(self.DRONE_IDS[drone_id+1:]) + self.OBSTACLE_IDS:  # sensor proximity read drone
                list_cp = p.getClosestPoints(self.DRONE_IDS[drone_id], obst_id, max_sensor_dist, physicsClientId = self.CLIENT)            
                if(len(list_cp) != 0): # there is obstacle near drone
                    for cp in list_cp:
                        x_dr, y_dr, z_dr = drone_state[0:3]
                        x_ob, y_ob, z_ob = cp[6]
                        theta = np.arctan2(y_ob - y_dr, x_ob - x_dr) * 180 / np.pi
                        eps = max_sensor_angle
                        if(-eps < theta < eps): #x+
                            obstacle_state[0] = 1
                        elif((90-eps) < theta < (90+eps)): #y+
                            obstacle_state[1] = 1
                        elif(-180 < theta <= (-180+eps) or (180-eps) < theta <= 180): #x-
                            obstacle_state[2] = 1
                        elif((-90-eps) < theta <(-90+eps)): #y-
                            obstacle_state[3] = 1

        elif(self.ACT_TYPE in [ActionType.XY_YAW, ActionType.XYZ_YAW]):
            obstacle_state = np.ones(4)
            drone_state = self._getDroneStateVector(drone_id)
            for obst_id in list(self.DRONE_IDS[:drone_id]) + list(self.DRONE_IDS[drone_id+1:]) + self.OBSTACLE_IDS:  # sensor proximity read drone
                list_cp = p.getClosestPoints(self.DRONE_IDS[drone_id], obst_id, max_sensor_dist, physicsClientId = self.CLIENT)
                if(len(list_cp) != 0): # there is obstacle near drone
                    for cp in list_cp:
                        x_dr, y_dr, z_dr = drone_state[0:3]
                        x_ob, y_ob, z_ob = cp[6]
                        dist = np.linalg.norm([x_ob - x_dr, y_ob - y_dr])
                        theta = np.arctan2(y_ob - y_dr, x_ob - x_dr) * 180 / np.pi
                        eps = max_sensor_angle
                        yaw = drone_state[9] * 180 / np.pi
                        if(-eps < theta - yaw < eps): 
                            obstacle_state[0] = np.clip(dist, 0, max_sensor_dist) / max_sensor_dist
                        elif((90-eps) < theta - yaw < (90+eps)): 
                            obstacle_state[1] = np.clip(dist, 0, max_sensor_dist) / max_sensor_dist
                        elif(-180 < theta - yaw <= (-180+eps) or (180-eps) < theta - yaw <= 180): 
                            obstacle_state[2] = np.clip(dist, 0, max_sensor_dist) / max_sensor_dist
                        elif((-90-eps) < theta - yaw <(-90+eps)):
                            obstacle_state[3] = np.clip(dist, 0, max_sensor_dist) / max_sensor_dist
        obstacle_state = np.ones(4)
        return obstacle_state

    def _isHitEverything(self, drone_ids):
        for i in range(len(drone_ids)):
            a, b = p.getAABB(drone_ids[i], physicsClientId = self.CLIENT) # Melihat batas posisi collision drone ke i
            list_obj = p.getOverlappingObjects(a, b, physicsClientId = self.CLIENT) # Melihat objek2 yang ada di batas posisi collision
            if(list_obj != None and len(list_obj) > 6): # 1 Quadcopter memiliki 6 link/bagian
                #print("Drone {}: Hit".format(i))
                return True
        return False

    def _isDroneTooFar(self, drone_ids, max_dist = None):
        # Looping untuk setiap pair of drone
        if(max_dist == None):
            max_dist = self.MAX_DISTANCE_BETWEEN_DRONE
        for i in range(len(drone_ids)):
            for j in range(i + 1, len(drone_ids)):
                dr1_states = self._getDroneStateVector(i)
                dr2_states = self._getDroneStateVector(j)
                dist = np.linalg.norm(dr1_states[0:3] - dr2_states[0:3])
                if (dist > max_dist):
                    #print("Drone Distance Too Far")
                    return True
        return False

    def _getCentroid(self, drone_ids):
        centroid = np.array([0, 0, 0], dtype=np.float32)
        for i in range(len(drone_ids)):
            dr_states = self._getDroneStateVector(i)
            centroid += dr_states[0:3]
        centroid /= len(drone_ids)
        return centroid


    def _isArrive(self, drone_ids, tol = 0.1, dest = None):
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        # Menghitung centroid of points dari kumpulan drone
        if(dest == None):
            dest = self.DEST_POINT
        centroid = self._getCentroid(drone_ids)
        if(np.linalg.norm(centroid - dest) < tol):
            if (np.linalg.norm(states[0,2]-states[1,2])) < 0.01 :
                # print("Drone ALL: _isArrive")
                return True
        else:
            return False
              