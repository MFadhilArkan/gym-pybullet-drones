import math
import numpy as np
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE
import os
import pybullet as p
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.envs.multi_agent_rl.BaseMultiagentAviary import BaseMultiagentAviary
from gym_pybullet_drones.utils.utils import nnlsRPM
from experiments.learning import shared_constants
#X merah, Y hijau, Z biru


class PayloadCoop(BaseMultiagentAviary):
    def __init__(self,
                 dest_point: np.ndarray=np.array(shared_constants.DEST_POINT),
                 episode_len_sec: float=shared_constants.EPISODE_LEN_SEC,
                 max_distance_between_drone: float=shared_constants.MAX_DISTANCE_BETWEEN_DRONE,
                 max_xy: float=shared_constants.MAX_XY,
                 max_z: float=shared_constants.MAX_Z,
                 k_move: float = shared_constants.K_MOVE,
                 drone_model: DroneModel=DroneModel(shared_constants.DRONE_MODEL),
                 num_drones: int=shared_constants.NUM_DRONES,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics(shared_constants.PHYSICS),
                 freq: int=shared_constants.FREQ,
                 aggregate_phy_steps: int=shared_constants.AGGR_PHY_STEPS,
                 gui=False,
                 record=False, 
                 obs: ObservationType=ObservationType(shared_constants.OBS),
                 act: ActionType=ActionType(shared_constants.ACT),                 
                 ):
                 
        if(initial_xyzs == None):
            initial_xyzs = self._initPositionOnCircle(num_drones, r = shared_constants.INIT_RADIUS, z = dest_point[2])

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
        self.Z_CONST = dest_point[2]
        self.MAX_DISTANCE_BETWEEN_DRONE = max_distance_between_drone
        self.MAX_XY = max_xy
        self.MAX_Z = max_z
        self.DEST_POINT = dest_point
        self.OBSTACLE_IDS = []
        self.EPISODE_LEN_SEC = episode_len_sec
        self.K_MOVE = k_move
        self.TRAINING_PHASE = 0
        self.POS_TOL = shared_constants.IS_ARRIVE_POS_TOL
        self.VEL_TOL = shared_constants.IS_ARRIVE_VEL_TOL
        self.MAX_DISTANCE_TO_ORIGIN = np.sqrt(self.MAX_XY**2 + self.MAX_XY**2 + self.MAX_Z)
        # assert self.NUM_DRONES == 2, "NUM_DRONES is not 2"
    ################################################################################

    def _actionSpace(self):
        if(self.ACT_TYPE == ActionType.JOYSTICK):
            return spaces.Dict({i: spaces.Discrete(5) for i in range(self.NUM_DRONES)})

        if self.ACT_TYPE in [ActionType.RPM, ActionType.DYN, ActionType.VEL]:
            size = 4
        elif self.ACT_TYPE in [ActionType.PID]:
            size = 3
        elif self.ACT_TYPE in [ActionType.VEL_YAW, ActionType.XYZ_YAW]:
            size = 5
        else:
            print("[ERROR] in BaseMultiagentAviary._actionSpace()")
            exit()
        return spaces.Dict({i: spaces.Box(low=-1*np.ones(size),
                                          high=np.ones(size),
                                          dtype=np.float32
                                          ) for i in range(self.NUM_DRONES)})
    ################################################################################

    def _observationSpace(self):
        if self.OBS_TYPE in [ObservationType.KIN, ObservationType.PAYLOAD_ONE_SENSOR]:
            if self.OBS_TYPE == ObservationType.KIN:
                #(x,y,z, r,p,y, x_dot, y_dot, z_dot, r_dot,p_dot,y_dot, ob_x+,ob_y+,ob_x-,ob_y-, x_dest-x,y_dest-y,z_dest-z,  x_dr2-x,y_dr2-y,z_dr2-z .....)
                #12 + 4 state obstacle, 3 state distance to dest_point, 3N state distance between drone,
                dist_drone = np.ones((3*(self.NUM_DRONES - 1)))
                low = np.array([-1,-1,0, -1,-1,-1, -1,-1,-1, -1,-1,-1, 0, -1,-1,-1])
                high = np.array([1,1,1,   1,1,1,    1,1,1,    1,1,1,   1,  1,1,1])

            elif self.OBS_TYPE == ObservationType.PAYLOAD_ONE_SENSOR:
                #1 state yaw, 1 state obstacle, 3 state distance to dest_point, 3N state distance between drone,
                dist_drone = np.ones((3*(self.NUM_DRONES - 1)))
                low = np.array([-1, 0, -1,-1,-1])
                high = np.array([1, 1,  1,1,1])

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
        if self.OBS_TYPE in [ObservationType.KIN, ObservationType.PAYLOAD_ONE_SENSOR]:
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
                mask = np.ones(3*(self.NUM_DRONES - 1))
                mask = np.hstack([[True]*12, True, [False]*3, [True]*3, mask])     
                return {i: obs_all[i, mask != 0] for i in range(self.NUM_DRONES)}

            elif self.OBS_TYPE == ObservationType.PAYLOAD_ONE_SENSOR:
                index_true = np.array([5, 12, 16, 17, 18, 19, 20, 21])
                mask = np.zeros(22)
                mask[index_true] = 1   
                return {i: obs_all[i, mask != 0] for i in range(self.NUM_DRONES)}
        else:
            print("[ERROR] in PayloadCoop._computeObs()")
            
    def _computeReward(self, dr_state_bef):
        drone_ids = self.getDroneIds()
        dr_state_aft = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)]) 
        
        reward = 0.0
        rwd_hit = shared_constants.RWD_HIT
        rwd_toofar_drone = shared_constants.RWD_TOOFAR_DRONE
        rwd_arrive = shared_constants.RWD_ARRIVE
        rwd_dist_dest = shared_constants.RWD_DIST_DEST
        rwd_dist_betw_drone = shared_constants.RWD_DIST_BETW_DRONE
        rwd_time = shared_constants.RWD_TIME
        rwd_energy = shared_constants.RWD_ENERGY

        # done = False
        # step_to_go = (self.EPISODE_LEN_SEC * self.SIM_FREQ - self.step_counter)
        
        if(self._isDroneTooFar(drone_ids)):
            reward += rwd_toofar_drone   
            done = True

        if(self._isArrive(drone_ids)):
            reward += rwd_arrive
            done = True

        # Time reward
        reward += rwd_time
    
        # Approaching dest
        cent_bef = dr_state_bef[:, 0:3].mean(axis = 0)
        cent_aft = dr_state_aft[:, 0:3].mean(axis = 0)
        dist_dest_diff = np.linalg.norm(cent_bef - self.DEST_POINT) - np.linalg.norm(cent_aft - self.DEST_POINT) # 0.001an
        reward += rwd_dist_dest * dist_dest_diff

        rewards = {i: reward for i in range(len(drone_ids))}      
        for i in range(len(drone_ids)):

            # Energy usage
            # rewards[i] += rwd_rpm * np.sum(dr_state_aft[16:20]) / 4
            rewards[i] += rwd_energy * np.linalg.norm(dr_state_aft[i][0:3] - dr_state_bef[i][0:3])

            # Distance between drone
            dist_drone_bef = np.linalg.norm(dr_state_bef[i][0:3] - dr_state_bef[(i+1)%2][0:3])
            dist_drone_aft = np.linalg.norm(dr_state_aft[i][0:3] - dr_state_bef[(i+1)%2][0:3])
            center = self.MAX_DISTANCE_BETWEEN_DRONE/2
            dist_drone_diff = np.sqrt((dist_drone_bef - center)**2) - np.sqrt((dist_drone_aft - center)**2)
            rewards[i] += rwd_dist_betw_drone * dist_drone_diff

            # isHitEverything
            if(self._isHitEverything(drone_ids[i])):
                rewards[i] += rwd_hit
        
        return rewards

    ################################################################################
    
    def _computeDone(self):
        drone_ids = self.getDroneIds()
        bool_val = self._isArrive(drone_ids) or self._isDroneTooFar(drone_ids) or (self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC) or self._isOutOfField(drone_ids)
        done = {i: bool_val for i in range(self.NUM_DRONES)}
        for i in range(len(drone_ids)):
            done[i] = done[i] or self._isHitEverything(drone_ids[i])
        done["__all__"] = True if True in done.values() else False
        return done

    def step(self,
             action
             ):
        """Advances the environment by one simulation step.

        Parameters
        ----------
        action : ndarray | dict[..]
            The input action for one or more drones, translated into RPMs by
            the specific implementation of `_preprocessAction()` in each subclass.

        Returns
        -------
        ndarray | dict[..]
            The step's observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        float | dict[..]
            The step's reward value(s), check the specific implementation of `_computeReward()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current epoisode is over, check the specific implementation of `_computeDone()`
            in each subclass for its format.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """
        #### Save PNG video frames if RECORD=True and GUI=False ####
        if self.RECORD and not self.GUI and self.step_counter%self.CAPTURE_FREQ == 0:
            [w, h, rgb, dep, seg] = p.getCameraImage(width=self.VID_WIDTH,
                                                     height=self.VID_HEIGHT,
                                                     shadow=1,
                                                     viewMatrix=self.CAM_VIEW,
                                                     projectionMatrix=self.CAM_PRO,
                                                     renderer=p.ER_TINY_RENDERER,
                                                     flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                                     physicsClientId=self.CLIENT
                                                     )
            (Image.fromarray(np.reshape(rgb, (h, w, 4)), 'RGBA')).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
            #### Save the depth or segmentation view instead #######
            # dep = ((dep-np.min(dep)) * 255 / (np.max(dep)-np.min(dep))).astype('uint8')
            # (Image.fromarray(np.reshape(dep, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
            # seg = ((seg-np.min(seg)) * 255 / (np.max(seg)-np.min(seg))).astype('uint8')
            # (Image.fromarray(np.reshape(seg, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
            self.FRAME_NUM += 1

        #### Read the GUI's input parameters #######################
        if self.GUI and self.USER_DEBUG:
            current_input_switch = p.readUserDebugParameter(self.INPUT_SWITCH, physicsClientId=self.CLIENT)
            if current_input_switch > self.last_input_switch:
                self.last_input_switch = current_input_switch
                self.USE_GUI_RPM = True if self.USE_GUI_RPM == False else False
        if self.USE_GUI_RPM:
            for i in range(4):
                self.gui_input[i] = p.readUserDebugParameter(int(self.SLIDERS[i]), physicsClientId=self.CLIENT)
            clipped_action = np.tile(self.gui_input, (self.NUM_DRONES, 1))
            if self.step_counter%(self.SIM_FREQ/2) == 0:
                self.GUI_INPUT_TEXT = [p.addUserDebugText("Using GUI RPM",
                                                          textPosition=[0, 0, 0],
                                                          textColorRGB=[1, 0, 0],
                                                          lifeTime=1,
                                                          textSize=2,
                                                          parentObjectUniqueId=self.DRONE_IDS[i],
                                                          parentLinkIndex=-1,
                                                          replaceItemUniqueId=int(self.GUI_INPUT_TEXT[i]),
                                                          physicsClientId=self.CLIENT
                                                          ) for i in range(self.NUM_DRONES)]
        #### Save, preprocess, and clip the action to the max. RPM #
        else:
            pass
            # clipped_action = np.reshape(self._preprocessAction(action), (self.NUM_DRONES, 4))
        #### Repeat for as many as the aggregate physics steps #####
        dr_state_bef = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        for _ in range(self.AGGR_PHY_STEPS):
            #### Update and store the drones kinematic info for certain
            #### Between aggregate steps for certain types of update ###
            if self.AGGR_PHY_STEPS > 1 and self.PHYSICS in [Physics.DYN, Physics.PYB_GND, Physics.PYB_DRAG, Physics.PYB_DW, Physics.PYB_GND_DRAG_DW]:
                self._updateAndStoreKinematicInformation()
                self._saveLastAction(action)
            #### Step the simulation using the desired physics update ##
            clipped_action = np.reshape(self._preprocessAction(action), (self.NUM_DRONES, 4))
            print(clipped_action[0,:])
            for i in range (self.NUM_DRONES):
                if self.PHYSICS == Physics.PYB:
                    self._physics(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.DYN:
                    self._dynamics(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_GND:
                    self._physics(clipped_action[i, :], i)
                    self._groundEffect(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_DRAG:
                    self._physics(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                elif self.PHYSICS == Physics.PYB_DW:
                    self._physics(clipped_action[i, :], i)
                    self._downwash(i)
                elif self.PHYSICS == Physics.PYB_GND_DRAG_DW:
                    self._physics(clipped_action[i, :], i)
                    self._groundEffect(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                    self._downwash(i)
            #### PyBullet computes the new state, unless Physics.DYN ###
            if self.PHYSICS != Physics.DYN:
                p.stepSimulation(physicsClientId=self.CLIENT)
            #### Save the last applied action (e.g. to compute drag) ###
            self.last_clipped_action = clipped_action
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Prepare the return values #############################
        obs = self._computeObs()
        reward = self._computeReward(dr_state_bef)
        done = self._computeDone()
        info = self._computeInfo()
        #### Advance the step counter ##############################
        self.step_counter = self.step_counter + (1 * self.AGGR_PHY_STEPS)
        return obs, reward, done, info

    ################################################################################
    
    def _computeInfo(self):
        return {i: {} for i in range(self.NUM_DRONES)}

    ################################################################################

    def _preprocessAction(self, action):
        self.ACTION_NOW = action
        K_MOVE = self.K_MOVE  
        rpm = np.zeros((self.NUM_DRONES,4))
        for k, v in action.items():
            if self.ACT_TYPE == ActionType.JOYSTICK:
                state = self._getDroneStateVector(int(k))
                if v == 0: # Arah X pos
                    target_pos = state[0:3] + K_MOVE * np.array([1, 0, 0])
                elif v == 1: # Arah Y pos
                    target_pos = state[0:3] + K_MOVE * np.array([0, 1, 0])
                elif v == 2: # Arah X neg
                    target_pos = state[0:3] + K_MOVE * np.array([-1, 0, 0])
                elif v == 3: # Arah Y neg
                    target_pos = state[0:3] + K_MOVE * np.array([0, -1, 0])
                elif v == 4: # Diam
                    target_pos = state[0:3]
                else:
                    target_pos = state[0:3]
                    print("Aksi tidak diketahui, drone ke-{} akan diam\n".format(k))
                rpm_k, _, _ = self.ctrl[int(k)].computeControl(control_timestep=self.AGGR_PHY_STEPS*self.TIMESTEP, 
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=np.array([target_pos[0], target_pos[1], self.Z_CONST])
                                                        )
                rpm[int(k),:] = rpm_k
            elif self.ACT_TYPE == ActionType.XYZ_YAW:
                state = self._getDroneStateVector(int(k))
                if np.linalg.norm(v[0:3]) != 0:
                    p_unit_vector = v[0:3] / np.linalg.norm(v[0:3])
                else:
                    p_unit_vector = np.zeros(3)
                delta_pos_limit = self.K_MOVE * self.MAX_SPEED_KMH * (1000/3600) * self.AGGR_PHY_STEPS/self.SIM_FREQ
                target_delta_pos = delta_pos_limit * np.abs(v[3]) * p_unit_vector
                target_pos = state[0:3] + target_delta_pos
                target_rpy = state[7:10] + np.pi / 4 * np.hstack([0,0,v[4]])
                rpm_k, _, _ = self.ctrl[int(k)].computeControl(control_timestep=self.AGGR_PHY_STEPS*self.TIMESTEP, 
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=target_pos,
                                                        target_rpy=target_rpy
                                                        )
                rpm[int(k),:] = rpm_k
            elif self.ACT_TYPE == ActionType.VEL_YAW:
                state = self._getDroneStateVector(int(k))
                if np.linalg.norm(v[0:3]) != 0:
                    v_unit_vector = v[0:3] / np.linalg.norm(v[0:3])
                else:
                    v_unit_vector = np.zeros(3)
                speed_limit = self.K_MOVE * self.MAX_SPEED_KMH * (1000/3600)
                target_vel = speed_limit * np.abs(v[3]) * v_unit_vector 
                target_rpy_rates =  np.array([0, 0, speed_limit * v[4]])
                
                rpm_k, _, _ = self.ctrl[int(k)].computeControl(control_timestep=self.AGGR_PHY_STEPS*self.TIMESTEP, 
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=state[0:3],
                                                        target_rpy=state[7:10], # keep current yaw
                                                        target_vel=target_vel, # target the desired velocity vector
                                                        target_rpy_rates= target_rpy_rates
                                                        )
                rpm[int(k),:] = rpm_k
            else:
                print("[ERROR] in BaseMultiagentAviary._preprocessAction()")
                exit()

            # self.TARGET_HISTORY[int(k), self.step_counter, :] = target_pos
            # self.POSITION_HISTORY[int(k), self.step_counter, :] = state[0:3] 
        return rpm

    ################################################################################

    def _clipAndNormalizeState(self,
                               state
                               ):
        STATE_TOL = shared_constants.STATE_TOL
        MAX_LIN_VEL_XY = self.MAX_SPEED_KMH /3.6 * STATE_TOL
        MAX_LIN_VEL_Z = self.MAX_SPEED_KMH /3.6 * STATE_TOL
        MAX_ANG_VEL = self.MAX_SPEED_KMH /3.6 * STATE_TOL * self.COLLISION_R
        MAX_XY = self.MAX_XY * STATE_TOL
        MAX_Z = self.MAX_Z * STATE_TOL
        MAX_DIST_GOAL_XY = self.MAX_XY
        MAX_DIST_GOAL_Z = self.MAX_Z
        MAX_DISTANCE_BETWEEN_DRONE = self.MAX_DISTANCE_BETWEEN_DRONE * STATE_TOL
        MAX_PITCH_ROLL = np.pi # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[3:5], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[6:8], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[8], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)
        clipped_ang_vel = np.clip(state[9:12], -MAX_ANG_VEL, MAX_ANG_VEL)
        clipped_dist_goal_xy = np.clip(state[16:18], -MAX_DIST_GOAL_XY, MAX_DIST_GOAL_XY)
        clipped_dist_goal_z = np.clip(state[18], -MAX_DIST_GOAL_Z, MAX_DIST_GOAL_Z)
        clipped_dist_drone = np.clip(state[19:], -MAX_DISTANCE_BETWEEN_DRONE, MAX_DISTANCE_BETWEEN_DRONE)
        
        state[0:2] = clipped_pos_xy / MAX_XY
        state[2] = clipped_pos_z / MAX_Z
        state[3:5] = clipped_rp / MAX_PITCH_ROLL
        state[5] = state[5] / np.pi # No reason to clip
        state[6:8] = clipped_vel_xy / MAX_LIN_VEL_XY
        state[8] = clipped_vel_z / MAX_LIN_VEL_XY
        state[9:12] = clipped_ang_vel / MAX_ANG_VEL
        state[16:18] = clipped_dist_goal_xy / MAX_DIST_GOAL_XY
        state[18] = clipped_dist_goal_z / MAX_DIST_GOAL_Z
        state[19:]= clipped_dist_drone / MAX_DISTANCE_BETWEEN_DRONE                                       
        return state
        
    ################################################################################

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

    ################################################################################

    def _isHitEverything(self, drone_id):
        a, b = p.getAABB(drone_id, physicsClientId = self.CLIENT) # Melihat batas posisi collision drone ke i
        list_obj = p.getOverlappingObjects(a, b, physicsClientId = self.CLIENT) # Melihat objek2 yang ada di batas posisi collision
        if(list_obj != None and len(list_obj) > 6): # 1 Quadcopter memiliki 6 link/bagian
            # print("Drone {}: _isHitEverything".format(drone_id))
            return True
        return False

    ################################################################################

    def _isDroneTooFar(self, drone_ids, max_dist_drone = None):
        # Looping untuk setiap pair of drone
        if(max_dist_drone == None):
            max_dist_drone = self.MAX_DISTANCE_BETWEEN_DRONE
        for i in range(len(drone_ids)):
            for j in range(i + 1, len(drone_ids)):
                dr1_states = self._getDroneStateVector(i)
                dr2_states = self._getDroneStateVector(j)
                dist = np.linalg.norm(dr1_states[0:3] - dr2_states[0:3])
                if (dist > max_dist_drone):
                    # print("Drone {}-{}: _isDroneTooFar".format(i, j))
                    return True
        return False

    def _isOutOfField(self, drone_ids):
        centroid = self._getCentroid(drone_ids)
        if(np.linalg.norm(centroid) > self.MAX_DISTANCE_TO_ORIGIN):
            # print("out of Field")
            return True
        else:
            return False

    def _getCentroid(self, drone_ids):
        centroid = np.array([0, 0, 0], dtype=np.float32)
        for i in range(len(drone_ids)):
            dr_states = self._getDroneStateVector(i)
            centroid += dr_states[0:3]
        centroid /= len(drone_ids)
        return centroid

    def _isArrive(self, drone_ids, dest = None):

        if(dest == None):
            dest = self.DEST_POINT
        
        #check if all drone is not moving
        for i in range(len(drone_ids)):
            dr_states = self._getDroneStateVector(i)
            if((dr_states[10:16] > self.VEL_TOL).any()):
                return False

        centroid = self._getCentroid(drone_ids)
        if(np.linalg.norm(centroid - dest) < self.POS_TOL):
            # print("Drone ALL: _isArrive")
            return True
        else:
            return False

    ################################################################################

    def _isObstacleNear(self, drone_id, max_sensor_dist = shared_constants.MAX_SENSOR_DIST, 
                        max_sensor_angle = shared_constants.MAX_SENSOR_ANGLE): 
        obstacle_state = np.ones(4)
        drone_state = self._getDroneStateVector(drone_id)
        for obst_id in list(self.DRONE_IDS[:drone_id]) + list(self.DRONE_IDS[drone_id+1:]) + self.OBSTACLE_IDS:
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
        return obstacle_state


        ################################################################################

    def _addObstaclesAt(self, position, orientation = [0, 0, 0], name = "cube_no_rotation.urdf"):
        id_ = p.loadURDF(name,
                    position,
                    p.getQuaternionFromEuler(np.array(orientation)),
                    physicsClientId=self.CLIENT
                    )
        self.OBSTACLE_IDS.append(id_) 

    def _addObstaclesAll(self):
        p_obst = np.random.uniform(shared_constants.MIN_DIST_FROM_ORIGIN, shared_constants.MAX_DIST_FROM_ORIGIN) * np.array(self.DEST_POINT) / np.linalg.norm((self.DEST_POINT))
        or_obst = [0, 0, np.random.uniform(0, 2*np.pi)]
        # self._addObstaclesAt([p_obst[0], p_obst[1], 1], or_obst, "cylinder.urdf")
        self._addObstaclesAt([p_obst[0], p_obst[1], 0.5], or_obst, "cube_no_rotation.urdf")
        self._addObstaclesAt(self.DEST_POINT, name = 'duck_vhacd.urdf')

    def _resetDestPoint(self):
        # r = np.random.uniform(0.5, 1.5) * np.linalg.norm(self.DEST_POINT[0:2])
        r = np.linalg.norm(self.DEST_POINT[0:2])
        t = np.random.uniform(0, 2*np.pi)
        self.DEST_POINT = [r * np.cos(t), r * np.sin(t), self.Z_CONST]

    def _initPositionOnCircle(self, n_drone, r = None, z = None, random = True):
        if(r == None):
            r = shared_constants.INIT_RADIUS
        if(z == None):
            z = self.Z_CONST
        ps = np.zeros((n_drone, 3))
        t0 = np.random.uniform(0, 2*np.pi)
        # t0 = 0
        for i in range(n_drone):
            x = r * np.cos((i*2*np.pi+t0)/n_drone)
            y = r * np.sin((i*2*np.pi+t0)/n_drone)
            ps[i, :] = x, y, z
        return ps