import pybullet as p
import time
import pybullet_data
import numpy as np

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf", useMaximalCoordinates=False)
cubeId = p.loadURDF("cube_no_rotation.urdf", [0, 5,0], useMaximalCoordinates=False)
sphereId = p.loadURDF("cube_no_rotation.urdf", [0, 0, 0], useMaximalCoordinates=False)

collisionFilterGroup = 0
collisionFilterMask = 0
p.setCollisionFilterGroupMask(cubeId, -1, 1, 1)
p.setCollisionFilterGroupMask(sphereId, -1, 1, 1)
p.setCollisionFilterGroupMask(planeId, -1, 1, 1)

enableCollision = 1
# p.setCollisionFilterPair(planeId, cubeId, -1, -1, enableCollision)
# p.setCollisionFilterPair(planeId, cubeId2, -1, -1, enableCollision)

p.setRealTimeSimulation(1)
p.setGravity(0, 0, -10)

i = 0
while (p.isConnected()):
  if(i % 100) == 0:
    print(p.getClosestPoints(1,2,4), '\n')
  time.sleep(1. / 240.)
  p.setGravity(0, 0, -10)
  i += 1