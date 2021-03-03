AGGR_PHY_STEPS = 5
NUM_DRONES = 2
ENV = 'payloadcoop'
"""int: Aggregate PyBullet/physics steps within a single call to env.step()."""

# For PayloadCoop
PHYSICS = 'pyb'
DRONE_MODEL = 'cf2x'
OBS = 'payload'
ACT = 'xyz_yaw'

DEST_POINT = [0, 6, 0.5]
MAX_DISTANCE_BETWEEN_DRONE = 1.0
EPISODE_LEN_SEC = 60.0
FREQ = 100
STATE_TOL = 1.05 # Tolerance of state value against maximum state constant
IS_ARRIVE_POS_TOL = 0.1
IS_ARRIVE_VEL_TOL = 0.05

MAX_XY = 30.0
MAX_Z = 3.0
K_MOVE = 0.1
MAX_SENSOR_DIST = 2.0
MAX_SENSOR_ANGLE = 10.0
SENSOR_MODE = 1 # 0: sensor points in absolute frame, 1: sensor points in body frame

# Reward
RWD_HIT = -1e4
RWD_TOOFAR_DRONE = -1e3
RWD_ARRIVE = 1e4
RWD_DEST = -0.1 / FREQ * AGGR_PHY_STEPS
RWD_TIME = -0 / FREQ * AGGR_PHY_STEPS
RWD_RPM = -0 / FREQ * AGGR_PHY_STEPS
RWD_DIST_Z = 1 / FREQ * AGGR_PHY_STEPS

# Randomization Obstacle
# 0 < RadiusInit = MAX_DISTANCE_BETWEEN_DRONE/4 < MIN_DIST_FROM_ORIGIN < obstacle_position < MAX_DIST_FROM_ORIGIN < DEST_POINT
MAX_DIST_FROM_ORIGIN = 4 
MIN_DIST_FROM_ORIGIN = 2

