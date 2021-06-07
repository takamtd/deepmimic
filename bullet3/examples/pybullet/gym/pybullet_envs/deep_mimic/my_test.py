import pybullet as p
import time
import math
import numpy as np
import pybullet_data

ATLAS = 0
HUMANOID = 1

# base - COM
# [  0.53942525  -0.10623837 -28.17860735]
# [ 1.31891597e-03 -1.23358114e-17 -1.17485780e-02]

# COM
# [-5.39425249e-01  1.06238366e-01  1.25178607e+02]
# [-1.31891597e-01  1.23358114e-15  9.75748578e+01]

# base
# [0.0, 0.0, 0.97]
# [0.0, 0.0, 0.964]

def quatMul(q1, q2):
  return [
      q1[3] * q2[0] + q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1],
      q1[3] * q2[1] + q1[1] * q2[3] + q1[2] * q2[0] - q1[0] * q2[2],
      q1[3] * q2[2] + q1[2] * q2[3] + q1[0] * q2[1] - q1[1] * q2[0],
      q1[3] * q2[3] - q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2]
  ]

def computeCOMposVel(uid: int):
    """Compute center-of-mass position and velocity."""
    num_joints = p.getNumJoints(uid)
    jointIndices = range(num_joints)
    link_states = p.getLinkStates(uid, jointIndices, computeLinkVelocity=1)
    link_pos = np.array([s[0] for s in link_states])
    link_vel = np.array([s[-2] for s in link_states])
    tot_mass = 0.
    masses = []
    for j in jointIndices:
      mass_, *_ = p.getDynamicsInfo(uid, j)
      masses.append(mass_)
      tot_mass += mass_
    masses = np.asarray(masses)[:, None]
    com_pos = np.sum(masses * link_pos, axis=0) / tot_mass
    com_vel = np.sum(masses * link_vel, axis=0) / tot_mass
    return com_pos, com_vel

character = ATLAS
# character = HUMANOID
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
if character == ATLAS:
  humanoid = p.loadURDF("/home/jovyan/kaleido_deepmimic/bullet3/examples/pybullet/gym/pybullet_data/atlas/atlas_v4_with_multisense.urdf", [0,0,0.965])
elif character == HUMANOID:
  humanoid = p.loadURDF("/home/jovyan/kaleido_deepmimic/bullet3/examples/pybullet/gym/pybullet_data/humanoid/humanoid.urdf", [0,0,0.964],
                        globalScaling=0.25*0.96322/0.881243, useFixedBase=True,
                        flags=p.URDF_MAINTAIN_LINK_ORDER)
gravId = p.addUserDebugParameter("gravity", -10, 10, -0.7)
planeId = p.loadURDF("plane_implicit.urdf", [0, 0, 0], [0, 0, 0, 1], useMaximalCoordinates=True)
jointIds = []
paramIds = []
p.setPhysicsEngineParameter(numSolverIterations=10)
p.changeDynamics(humanoid, -1, linearDamping=0, angularDamping=0)
y2z = p.getQuaternionFromEuler([math.pi * 0.5, 0, 0])
z2y = p.getQuaternionFromEuler([-math.pi * 0.5, 0, 0])
for j in range(p.getNumJoints(humanoid)):
  p.changeDynamics(humanoid, j, linearDamping=0, angularDamping=0)
  info = p.getJointInfo(humanoid, j)
  #print(info)
  jointName = info[1]
  jointType = info[2]
  if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
    jointIds.append(j)
    if character == ATLAS:
      # paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), -4, 4, 0))
      if(j == 4):
        paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), -4, 4, -math.pi * 0.5))
      elif(j == 5 or j == 12 or j == 13):
        paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), -4, 4, math.pi * 0.5))
      else:
        paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), -4, 4, 0))
    elif character == HUMANOID:
      paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), -4, 4, 0))
  p.addUserDebugLine([0, 0, 0], [0.2, 0, 0], [1, 0, 0], parentObjectUniqueId = humanoid, parentLinkIndex = j)
  p.addUserDebugLine([0, 0, 0], [0, 0.2, 0], [0, 1, 0], parentObjectUniqueId = humanoid, parentLinkIndex = j)
  p.addUserDebugLine([0, 0, 0], [0, 0, 0.2], [0, 0, 1], parentObjectUniqueId = humanoid, parentLinkIndex = j)
  p.changeVisualShape(humanoid, j, rgbaColor=[1, 1, 1, 0.9])

# orn = p.getQuaternionFromEuler([math.pi/8, math.pi/6, -math.pi/4])
# roll_pitch_yaw = p.getEulerFromQuaternion(orn)
# orn1 = p.getQuaternionFromEuler([roll_pitch_yaw[0], roll_pitch_yaw[2], -roll_pitch_yaw[1]])
# orn2 = p.getQuaternionFromEuler([math.pi/8 + math.pi * 0.5, math.pi/6, -math.pi/4])
# root_orn_sim = quatMul(orn, y2z)

# print(root_orn_sim)
# print(orn2)

p.setRealTimeSimulation(1)
# if character == ATLAS:
#   orn = p.getQuaternionFromEuler([math.pi/8, math.pi/6, -math.pi/4])
#   # p.resetBasePositionAndOrientation(humanoid, [0, 0, 0.97], [0, 0, 0, 1])
#   p.resetBasePositionAndOrientation(humanoid, [0, 0, 0.97], orn)
# elif character == HUMANOID:
#   orn3 = p.getQuaternionFromEuler([math.pi/8 + math.pi * 0.5, math.pi/6, -math.pi/4])
#   p.resetBasePositionAndOrientation(humanoid, [0, 0, 0.964], root_orn_sim)
#   change_orn = p.getQuaternionFromEuler([0, 0, math.pi/2])
#   # change_orn = [0.0288280000,       -0.0587060000,       -0.0625510000, 0.9958970000]
#   base, base_vel = p.getBasePositionAndOrientation(humanoid)
  
  
# base, base_vel = p.getBasePositionAndOrientation(humanoid)
# COM, COMVel = computeCOMposVel(humanoid)
# print(base)
while (1):
  p.setGravity(0, 0, p.readUserDebugParameter(gravId))
  # p.resetJointStateMultiDof(humanoid, 3, change_orn, [0, 0, 0])
  # p.resetJointStateMultiDof(humanoid, 4, [0], [0])
  # p.resetJointStateMultiDof(humanoid, 5, p.getQuaternionFromEuler([0, 0, math.pi/4]), [0, 0, 0])
  for i in range(len(paramIds)):
    c = paramIds[i]
    targetPos = p.readUserDebugParameter(c)
    p.setJointMotorControl2(humanoid, jointIds[i], p.POSITION_CONTROL, targetPos, force=5 * 240.)
    # print(p.getJointState(humanoid, 14)[0])
    # print(p.getBasePositionAndOrientation(humanoid)[0])
