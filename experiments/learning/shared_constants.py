import numpy as np
AGGR_PHY_STEPS = 5
FREQ = 240
NUM_DRONES = 2
ENV = 'payloadcoop'

# For PayloadCoop
PHYSICS = 'pyb'
DRONE_MODEL = 'cf2x'
OBS = 'payload_one_sensor'
ACT = 'xyz_yaw'
DEST_POINT = np.array([0, 10, 0.5])
MAX_DISTANCE_BETWEEN_DRONE = 2.0
INIT_RADIUS = MAX_DISTANCE_BETWEEN_DRONE / 5
EPISODE_LEN_SEC = 25
STATE_TOL = 1.05 # Tolerance of state value against maximum state constant
IS_ARRIVE_POS_TOL = 0.25
IS_ARRIVE_VEL_TOL = 1000
MAX_XY = 15.0
MAX_Z = 3.0
K_MOVE = 0.5
MAX_SENSOR_DIST = 5.0
MAX_SENSOR_ANGLE = 30.0
# Randomization Obstacle
# 0 < RadiusInit = MAX_DISTANCE_BETWEEN_DRONE/4 < MIN_DIST_FROM_ORIGIN < obstacle_position < MAX_DIST_FROM_ORIGIN < DEST_POINT
MAX_DIST_FROM_ORIGIN = 4
MIN_DIST_FROM_ORIGIN = 2


# Reward
RWD_HIT = -1e2
RWD_TOOFAR_DRONE = -1e2
RWD_ARRIVE = 2e2
RWD_DIST_DEST = 0.5 / FREQ * AGGR_PHY_STEPS #Jika optimal (v maks), total rewardnya norm(dest) * k / Freq * Aggr_phy_step
RWD_DIST_BETW_DRONE = 1 / FREQ * AGGR_PHY_STEPS #Jika optimal, total return = 0
RWD_TIME = -0 / FREQ * AGGR_PHY_STEPS
RWD_ENERGY = -0.1 / FREQ * AGGR_PHY_STEPS 

# Training
LR = 5e-3
GAMMA = 0.99
N_STEP = 4

# assert 0 <= MAX_DISTANCE_BETWEEN_DRONE / 4 <= MIN_DIST_FROM_ORIGIN <= MAX_DIST_FROM_ORIGIN <= np.linalg.norm(DEST_POINT), "Error MIN/MAX_DIST_FROM_ORIGIN"

