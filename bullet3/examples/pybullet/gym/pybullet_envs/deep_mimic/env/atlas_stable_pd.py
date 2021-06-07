from pybullet_utils import pd_controller_stable
from pybullet_envs.deep_mimic.env import atlas_pose_interpolator
import math
import numpy as np
import pandas as pd
import os

chest = 1
neck = 2
rightHip = 3
rightKnee = 4
rightAnkle = 5
rightShoulder = 6
rightElbow = 7
leftHip = 9
leftKnee = 10
leftAnkle = 11
leftShoulder = 12
leftElbow = 13
jointFrictionForce = 0

back_bkz = 0
back_bky = 1
back_bkx = 2
l_arm_shz = 3
l_arm_shx = 4
l_arm_ely = 5
l_arm_elx = 6
l_arm_wry = 7
l_arm_wrx = 8
l_arm_wry2 = 9
neck_ry = 10
r_arm_shz = 11
r_arm_shx = 12
r_arm_ely = 13
r_arm_elx = 14
r_arm_wry = 15
r_arm_wrx = 16
r_arm_wry2 = 17
l_leg_hpz = 18
l_leg_hpx = 19
l_leg_hpy = 20
l_leg_kny = 21
l_leg_aky = 22
l_leg_akx = 23
r_leg_hpz = 24
r_leg_hpx = 25
r_leg_hpy = 26
r_leg_kny = 27
r_leg_aky = 28
r_leg_akx = 29

OFF = 0
ON = 1

class AtlasStablePD(object):

  def __init__( self, pybullet_client, mocap_data, timeStep, 
                useFixedBase=True, arg_parser=None, useComReward=True):
    self._pybullet_client = pybullet_client
    self._mocap_data = mocap_data
    self._arg_parser = arg_parser
    self._log_count = -4
    print("LOADING atlas!")
    self._z2y = self._pybullet_client.getQuaternionFromEuler([-math.pi * 0.5, 0, 0])
    self._y2z = self._pybullet_client.getQuaternionFromEuler([math.pi * 0.5, 0, 0])
    self._x2z = self._pybullet_client.getQuaternionFromEuler([0, -math.pi * 0.5, 0])
    #flags=self._pybullet_client.URDF_MAINTAIN_LINK_ORDER#+self._pybullet_client.URDF_USE_SELF_COLLISION+self._pybullet_client.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
    self._sim_model = self._pybullet_client.loadURDF(
        "atlas/atlas_v4_with_multisense.urdf", [0, 1.0, 0], self._z2y,
        # globalScaling=0.9,
        useFixedBase=useFixedBase,
        flags=self._pybullet_client.URDF_USE_SELF_COLLISION + self._pybullet_client.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
    #  + self._pybullet_client.URDF_MAINTAIN_LINK_ORDER
    #     useFixedBase=useFixedBase,
    #     flags=flags
    #self._pybullet_client.setCollisionFilterGroupMask(self._sim_model,-1,collisionFilterGroup=0,collisionFilterMask=0)
    #for j in range (self._pybullet_client.getNumJoints(self._sim_model)):
    #  self._pybullet_client.setCollisionFilterGroupMask(self._sim_model,j,collisionFilterGroup=0,collisionFilterMask=0)

    self._kin_end_effectors = [5, 8, 11, 14]  #ankle and wrist, both left and right
    self._end_effectors = [9, 17, 23, 29]
    # self._kin_end_effectors = [11, 5]
    # self._end_effectors = [23, 29]

    self._kin_model = self._pybullet_client.loadURDF(
        "atlas/atlas_v4_with_multisense.urdf", [0, 1.0, 0], self._z2y,
        # globalScaling=0.9,
        useFixedBase=useFixedBase)

    # self._kin_model = self._pybullet_client.loadURDF(
    #     "humanoid/humanoid.urdf", [0, 0.85, 0],
    #     # globalScaling=0.25,
    #     globalScaling=0.27325607125389934,
    #     useFixedBase=True,
    #     flags=self._pybullet_client.URDF_MAINTAIN_LINK_ORDER)

    self._pybullet_client.changeDynamics(self._sim_model, -1, lateralFriction=0.9)
    for j in range(self._pybullet_client.getNumJoints(self._sim_model)):
      self._pybullet_client.changeDynamics(self._sim_model, j, lateralFriction=0.9)

    self._pybullet_client.changeDynamics(self._sim_model, -1, linearDamping=0, angularDamping=0)
    self._pybullet_client.changeDynamics(self._kin_model, -1, linearDamping=0, angularDamping=0)

    #todo: add feature to disable simulation for a particular object. Until then, disable all collisions
    self._pybullet_client.setCollisionFilterGroupMask(self._kin_model,
                                                      -1,
                                                      collisionFilterGroup=0,
                                                      collisionFilterMask=0)
    self._pybullet_client.changeDynamics(
        self._kin_model,
        -1,
        activationState=self._pybullet_client.ACTIVATION_STATE_SLEEP +
        self._pybullet_client.ACTIVATION_STATE_ENABLE_SLEEPING +
        self._pybullet_client.ACTIVATION_STATE_DISABLE_WAKEUP)
    alpha = 0.4
    self._pybullet_client.changeVisualShape(self._kin_model, -1, rgbaColor=[1, 1, 1, alpha])
    for j in range(self._pybullet_client.getNumJoints(self._kin_model)):
      self._pybullet_client.setCollisionFilterGroupMask(self._kin_model,
                                                        j,
                                                        collisionFilterGroup=0,
                                                        collisionFilterMask=0)
      self._pybullet_client.changeDynamics(
          self._kin_model,
          j,
          activationState=self._pybullet_client.ACTIVATION_STATE_SLEEP +
          self._pybullet_client.ACTIVATION_STATE_ENABLE_SLEEPING +
          self._pybullet_client.ACTIVATION_STATE_DISABLE_WAKEUP)
      self._pybullet_client.changeVisualShape(self._kin_model, j, rgbaColor=[1, 1, 1, alpha])

    self._poseInterpolator = atlas_pose_interpolator.AtlasPoseInterpolator()

    for i in range(self._mocap_data.NumFrames() - 1):
      frameData = self._mocap_data._motion_data['Frames'][i]
      self._poseInterpolator.PostProcessMotionData(frameData)

    self._stablePD = pd_controller_stable.PDControllerStableMultiDof(self._pybullet_client)
    self._timeStep = timeStep
    self._kpOrg = [
        0, 0, 0, 0, 0, 0, 0,
        1000, 1000, 1000,
        400, 400, 400, 300, 100, 100, 100,
        100,
        400, 400, 400, 300, 100, 100, 100,
        500, 500, 500, 500, 400, 400,
        500, 500, 500, 500, 400, 400
    ]
    self._kdOrg = [
        0, 0, 0, 0, 0, 0, 0,
        100, 100, 100,
        40, 40, 40, 30, 10, 10, 10,
        10,
        40, 40, 40, 30, 10, 10, 10,
        50, 50, 50, 50, 40, 40,
        50, 50, 50, 50, 40, 40
    ]
    # self._kpOrg = [
    #     0, 0, 0, 0, 0, 0, 0, 
    #     1000, 1000, 1000, 1000,   #chest
    #     100, 100, 100, 100,       #neck
    #     500, 500, 500, 500,       #r_hip
    #     500,                      #r_knee
    #     400, 400, 400, 400,       #r_ankle
    #     400, 400, 400, 400,       #r_sholder
    #     300,                      #r_elbow
    #     500, 500, 500, 500, 
    #     500, 
    #     400, 400, 400, 400,
    #     400, 400, 400, 400, 
    #     300
    # ]
    # self._kdOrg = [
    #     0, 0, 0, 0, 0, 0, 0,      
    #     100, 100, 100, 100,       #chest
    #     10, 10, 10, 10,           #neck
    #     50, 50, 50, 50,           #r_hip
    #     50,                       #r_knee
    #     40, 40, 40, 40,           #r_ankle
    #     40, 40, 40, 40,           #r_sholder
    #     30,                       #r_elbow

    #     50, 50, 50, 50, 
    #     50, 
    #     40, 40, 40, 40, 
    #     40, 40, 40, 40, 
    #     30
    # ]

    self._kin_jointIndicesAll = [
        chest, neck, rightHip, rightKnee, rightAnkle, rightShoulder, rightElbow, leftHip, leftKnee,
        leftAnkle, leftShoulder, leftElbow
    ]

    self._jointIndicesAll = [
      back_bkz, back_bky, back_bkx,
      l_arm_shz, l_arm_shx,
      l_arm_ely, l_arm_elx,
      l_arm_wry, l_arm_wrx, l_arm_wry2,
      neck_ry,
      r_arm_shz, r_arm_shx,
      r_arm_ely, r_arm_elx,
      r_arm_wry, r_arm_wrx, r_arm_wry2,
      l_leg_hpz, l_leg_hpx, l_leg_hpy,
      l_leg_kny,
      l_leg_aky, l_leg_akx,
      r_leg_hpz, r_leg_hpx, r_leg_hpy,
      r_leg_kny,
      r_leg_aky, r_leg_akx
    ]

    for j in self._jointIndicesAll:
      #self._pybullet_client.setJointMotorControlMultiDof(self._sim_model, j, self._pybullet_client.POSITION_CONTROL, force=[1,1,1])
      self._pybullet_client.setJointMotorControl2(self._sim_model,
                                                  j,
                                                  self._pybullet_client.POSITION_CONTROL,
                                                  targetPosition=0,
                                                  positionGain=0,
                                                  targetVelocity=0,
                                                  force=jointFrictionForce)
      self._pybullet_client.setJointMotorControlMultiDof(
          self._sim_model,
          j,
          self._pybullet_client.POSITION_CONTROL,
          targetPosition=[0, 0, 0, 1],
          targetVelocity=[0, 0, 0],
          positionGain=0,
          velocityGain=1,
          force=[jointFrictionForce, jointFrictionForce, jointFrictionForce])
    
    for j in self._jointIndicesAll:
      self._pybullet_client.setJointMotorControl2(self._kin_model,
                                                  j,
                                                  self._pybullet_client.POSITION_CONTROL,
                                                  targetPosition=0,
                                                  positionGain=0,
                                                  targetVelocity=0,
                                                  force=0)
      self._pybullet_client.setJointMotorControlMultiDof(
          self._kin_model,
          j,
          self._pybullet_client.POSITION_CONTROL,
          targetPosition=[0, 0, 0, 1],
          targetVelocity=[0, 0, 0],
          positionGain=0,
          velocityGain=1,
          force=[jointFrictionForce, jointFrictionForce, 0])

    self._jointDofCounts = [1 for j in range(len(self._jointIndicesAll))]
    self._kin_jointDofCounts = [4, 4, 4, 1, 4, 4, 1, 4, 1, 4, 4, 1]
    # self._kin_jointDofCounts = [4, 4, 4, 4, 1, 4, 4, 1, 4, 1, 4, 4, 1]

    #only those body parts/links are allowed to touch the ground, otherwise the episode terminates
    fall_contact_bodies = []
    if self._arg_parser is not None:
      fall_contact_bodies = self._arg_parser.parse_ints("fall_contact_bodies")
    self._fall_contact_body_parts = fall_contact_bodies

    #[x,y,z] base position and [x,y,z,w] base orientation!
    self._totalDofs = 7
    for dof in self._jointDofCounts:
      self._totalDofs += dof
    self.setSimTime(0)

    self._useComReward = useComReward
    self._count = 0
    self.resetPose()

  def resetPose(self):
    #print("resetPose with self._frame=", self._frame, " and self._frameFraction=",self._frameFraction)
    pose = self.computePose(self._frameFraction)
    self.initializePoseAtlas(self._poseInterpolator, self._kin_model, initBase=False)
    self.initializePoseAtlas(self._poseInterpolator, self._sim_model, initBase=True)
    self._log_mode = OFF
    if self._log_mode == ON:
      if self._log_count == -4:
        self._err_log_pathdir =        "/home/jovyan/kaleido_deepmimic/bullet3/examples/pybullet/gym/pybullet_envs/deep_mimic/logs/log6_01/data/err/"
        self._sim_log_pathdir =        "/home/jovyan/kaleido_deepmimic/bullet3/examples/pybullet/gym/pybullet_envs/deep_mimic/logs/log6_01/data/sim/"
        self._kin_log_pathdir =        "/home/jovyan/kaleido_deepmimic/bullet3/examples/pybullet/gym/pybullet_envs/deep_mimic/logs/log6_01/data/kin/"
        os.makedirs(self._err_log_pathdir)
        os.makedirs(self._sim_log_pathdir)
        os.makedirs(self._kin_log_pathdir)
      self._err_log_path = self._err_log_pathdir + "err_log{}.csv".format(self._log_count)
      self._sim_log_path = self._sim_log_pathdir + "sim_log{}.csv".format(self._log_count)
      self._kin_log_path = self._kin_log_pathdir + "kin_log{}.csv".format(self._log_count)
      self._log_count += 1
      self._log_columns = ["root_pos_x", "root_pos_y", "root_pos_z", "root_vel_x", "root_vel_y", "root_vel_z",
                            "root_orn", "root_orn_r", "root_orn_p", "root_orn_y", "root_ang_vel_x", "root_ang_vel_y", "root_ang_vel_z",
                            "com_pos_x", "com_pos_y", "com_pos_z", "com_vel_x", "com_vel_y", "com_vel_z",
                            "back_bkz_angle", "back_bkz_vel", "back_bky_angle", "back_bky_vel", "back_bkx_angle", "back_bkx_vel",
                            "neck_ry_angle", "neck_ry_vel",
                            "r_leg_hpz_angle", "r_leg_hpz_vel", "r_leg_hpx_angle", "r_leg_hpx_vel", "r_leg_hpy_angle", "r_leg_hpy_vel",
                            "r_leg_kny_angle", "r_leg_kny_vel",
                            "r_leg_aky_angle", "r_leg_aky_vel", "r_leg_akx_angle", "r_leg_akx_vel",
                            "r_arm_shz_angle", "r_arm_shz_vel", "r_arm_shx_angle", "r_arm_shx_vel", "r_arm_ely_angle", "r_arm_ely_vel",
                            "r_arm_elx_angle", "r_arm_elx_vel",
                            "l_leg_hpz_angle", "l_leg_hpz_vel", "l_leg_hpx_angle", "l_leg_hpx_vel", "l_leg_hpy_angle", "l_leg_hpy_vel",
                            "l_leg_kny_angle", "l_leg_kny_vel",
                            "l_leg_aky_angle", "l_leg_aky_vel", "l_leg_akx_angle", "l_leg_akx_vel",
                            "l_arm_shz_angle", "l_arm_shz_vel", "l_arm_shx_angle", "l_arm_shx_vel", "l_arm_ely_angle", "l_arm_ely_vel",
                            "l_arm_elx_angle", "l_arm_elx_vel",
                            "l_arm_wry2_pos_x", "l_arm_wry2_pos_y", "l_arm_wry2_pos_z",
                            "r_arm_wry2_pos_x", "r_arm_wry2_pos_y", "r_arm_wry2_pos_z",
                            "l_leg_akx_pos_x", "l_leg_akx_pos_y", "l_leg_akx_pos_z",
                            "r_leg_akx_pos_x", "r_leg_akx_pos_y", "r_leg_akx_pos_z"]
      
      self._err_log_df = pd.DataFrame(index=[], columns=self._log_columns)
      self._sim_log_df = pd.DataFrame(index=[], columns=self._log_columns)
      self._kin_log_df = pd.DataFrame(index=[], columns=self._log_columns)

  def initializePoseAtlas(self, pose, phys_model, initBase, initializeVelocity=True):
    useArray = True
    if initializeVelocity:
      if initBase:
        base_pose = pose._basePos
        base_pose[1] += 0.1
        # base_pose[0] += -0.8
        base_orn = self.quatMul(pose._baseOrn, self._z2y)
        # base_orn = self.quatMul(self._x2z, base_orn)
        self._pybullet_client.resetBasePositionAndOrientation(phys_model, base_pose, base_orn)
        # base_orn = self.quatMul(self._z2y, [0, 0, 0, 1])
        # self._pybullet_client.resetBasePositionAndOrientation(phys_model, [0, 0.95, 0], [0, 0, 0, 1])
        # self._pybullet_client.resetBaseVelocity(phys_model, pose._baseLinVel, pose._baseAngVel)
        base_ang_vel = [
          pose._baseAngVel[0], -pose._baseAngVel[2], pose._baseAngVel[1]
        ]
        self._pybullet_client.resetBaseVelocity(phys_model, pose._baseLinVel, base_ang_vel)
      if useArray:
        # for index in range(len(self._jointIndicesAll)):
        #   self._pybullet_client.resetJointState(phys_model, index, 0.0, 0.0)
        
        # self._pybullet_client.resetJointState(phys_model, l_arm_shx, -1.5708, 0.0)
        # self._pybullet_client.resetJointState(phys_model, r_arm_shx, 1.5708, 0.0)
        # self._pybullet_client.resetJointState(phys_model, l_arm_ely, 1.5708, 0.0)
        # self._pybullet_client.resetJointState(phys_model, r_arm_ely, 1.5708, 0.0)

        self._pybullet_client.resetJointState(phys_model, neck_ry, pose._neckRot[2], 0.0)
        self._pybullet_client.resetJointState(phys_model, l_arm_elx, pose._leftElbowRot[0], pose._leftElbowVel[0])
        self._pybullet_client.resetJointState(phys_model, r_arm_elx, -pose._rightElbowRot[0], -pose._rightElbowVel[0])
        self._pybullet_client.resetJointState(phys_model, l_leg_kny, -pose._leftKneeRot[0], -pose._leftKneeVel[0])
        self._pybullet_client.resetJointState(phys_model, r_leg_kny, -pose._rightKneeRot[0], -pose._rightKneeVel[0])
        # l_leg_hp
        roll_pitch_yaw = self._pybullet_client.getEulerFromQuaternion(pose._leftHipRot)
        self._pybullet_client.resetJointState(phys_model, l_leg_hpz, roll_pitch_yaw[1], pose._leftHipVel[1])
        self._pybullet_client.resetJointState(phys_model, l_leg_hpy, -roll_pitch_yaw[2], -pose._leftHipVel[2])
        self._pybullet_client.resetJointState(phys_model, l_leg_hpx, roll_pitch_yaw[0], pose._leftHipVel[0])
        # r_leg_hp
        roll_pitch_yaw = self._pybullet_client.getEulerFromQuaternion(pose._rightHipRot)
        self._pybullet_client.resetJointState(phys_model, r_leg_hpz, roll_pitch_yaw[1], pose._rightHipVel[1])
        self._pybullet_client.resetJointState(phys_model, r_leg_hpy, -roll_pitch_yaw[2], -pose._rightHipVel[2])
        self._pybullet_client.resetJointState(phys_model, r_leg_hpx, roll_pitch_yaw[0], pose._rightHipVel[0])
        # back_bk
        roll_pitch_yaw = self._pybullet_client.getEulerFromQuaternion(pose._chestRot)
        self._pybullet_client.resetJointState(phys_model, back_bkz, roll_pitch_yaw[1], pose._chestVel[1])
        self._pybullet_client.resetJointState(phys_model, back_bky, -roll_pitch_yaw[2], -pose._chestVel[2])
        self._pybullet_client.resetJointState(phys_model, back_bkx, roll_pitch_yaw[0], pose._chestVel[0])
        # # l_leg_aky
        roll_pitch_yaw = self._pybullet_client.getEulerFromQuaternion(pose._leftAnkleRot)
        self._pybullet_client.resetJointState(phys_model, l_leg_aky, -roll_pitch_yaw[2], -pose._leftAnkleVel[2])
        self._pybullet_client.resetJointState(phys_model, l_leg_akx, roll_pitch_yaw[0], pose._leftAnkleVel[0])
        # # r_leg_aky
        roll_pitch_yaw = self._pybullet_client.getEulerFromQuaternion(pose._leftAnkleRot)
        self._pybullet_client.resetJointState(phys_model, r_leg_aky, -roll_pitch_yaw[2], -pose._rightAnkleVel[2])
        self._pybullet_client.resetJointState(phys_model, r_leg_akx, roll_pitch_yaw[0], pose._rightAnkleVel[0])
        # l_arm_sh
        roll_pitch_yaw = self._pybullet_client.getEulerFromQuaternion(pose._leftShoulderRot)
        self._pybullet_client.resetJointState(phys_model, l_arm_shz, roll_pitch_yaw[1], pose._leftShoulderVel[1])
        self._pybullet_client.resetJointState(phys_model, l_arm_ely, math.pi/2, 0)
        self._pybullet_client.resetJointState(phys_model, l_arm_shx, roll_pitch_yaw[0]-math.pi/2, pose._leftShoulderVel[0])
        # r_arm_sh
        roll_pitch_yaw = self._pybullet_client.getEulerFromQuaternion(pose._rightShoulderRot)
        self._pybullet_client.resetJointState(phys_model, r_arm_shz, roll_pitch_yaw[1], pose._rightShoulderVel[1])
        self._pybullet_client.resetJointState(phys_model, r_arm_ely, math.pi/2, 0)
        self._pybullet_client.resetJointState(phys_model, r_arm_shx, roll_pitch_yaw[0]+math.pi/2, pose._rightShoulderVel[0])



        # print("***base***")
        # print("sim_model")
        # print(self._pybullet_client.getBasePositionAndOrientation(self._sim_model))
        # orn = self._pybullet_client.getBasePositionAndOrientation(self._sim_model)[1]
        # print(self._pybullet_client.getAxisAngleFromQuaternion(orn))

        # print("kin_model")
        # print(self._pybullet_client.getBasePositionAndOrientation(self._kin_model))
        # print(self._pybullet_client.getAxisAngleFromQuaternion(self._pybullet_client.getBasePositionAndOrientation(self._kin_model)[1]))
        # print("")

        # print("***back_bkz***")
        # print("sim_model")
        # sim_orn = self._pybullet_client.getLinkState(self._sim_model, neck_ry)[1]
        # print(sim_orn)
        # print(self._pybullet_client.getAxisAngleFromQuaternion(sim_orn))
        # changed_orn = self.quatMul(self._y2z, sim_orn)
        # print(changed_orn)
        # print(self._pybullet_client.getAxisAngleFromQuaternion(changed_orn))
        # print("kin_model")
        # kin_orn = self._pybullet_client.getLinkState(self._kin_model, neck)[1]
        # print(kin_orn)
        # print(self._pybullet_client.getAxisAngleFromQuaternion(kin_orn))
        # print(self._pybullet_client.getEulerFromQuaternion(kin_orn))
        # print("diff")
        # diff = self._pybullet_client.getDifferenceQuaternion(changed_orn, kin_orn)
        # print(diff)
        # print(self._pybullet_client.getAxisAngleFromQuaternion(diff))
        # print(self._pybullet_client.getAxisAngleFromQuaternion(diff)[1]*180/math.pi)
        # print("")

        # root_orn_sim = self.quatMul(self._y2z, rootOrnSim)
        
        # if(self._count >= 1):
        #   while True:
        #     pass
        # self._count += 1

        # zeros = [0 for i in range(30)]
        # indices = [j for j in range(30)]
        # jointPositions = zeros
        # jointVelocities = zeros
        # self._pybullet_client.resetJointStatesMultiDof(phys_model, indices, jointPositions, jointVelocities)

  def initializePose(self, pose, phys_model, initBase, initializeVelocity=True):
    
    useArray = True
    if initializeVelocity:
      if initBase:
        # self._pybullet_client.resetBasePositionAndOrientation(phys_model, pose._basePos, pose._baseOrn)
        # self._pybullet_client.resetBaseVelocity(phys_model, pose._baseLinVel, pose._baseAngVel)
        base_pose = pose._basePos
        base_pose[1] += 0.082
        self._pybullet_client.resetBasePositionAndOrientation(phys_model, base_pose, pose._baseOrn)
        self._pybullet_client.resetBaseVelocity(phys_model, pose._baseLinVel, pose._baseAngVel)
        # base_orn = self.quatMul(self._x2z, pose._baseOrn)
        # self._pybullet_client.resetBasePositionAndOrientation(phys_model, pose._basePos, base_orn)
      if useArray:
        # base_orn = self.quatMul(self._x2z, pose._baseOrn)
        # self._pybullet_client.resetBasePositionAndOrientation(phys_model, pose._basePos, base_orn)
        indices = [chest,neck,rightHip,rightKnee,
                  rightAnkle, rightShoulder, rightElbow,leftHip,
                  leftKnee, leftAnkle, leftShoulder,leftElbow]
        jointPositions = [pose._chestRot, pose._neckRot, pose._rightHipRot, pose._rightKneeRot,
                          pose._rightAnkleRot, pose._rightShoulderRot, pose._rightElbowRot, pose._leftHipRot,
                          pose._leftKneeRot, pose._leftAnkleRot, pose._leftShoulderRot, pose._leftElbowRot]
        jointVelocities = [pose._chestVel, pose._neckVel, pose._rightHipVel, pose._rightKneeVel,
                          pose._rightAnkleVel, pose._rightShoulderVel, pose._rightElbowVel, pose._leftHipVel,
                          pose._leftKneeVel, pose._leftAnkleVel, pose._leftShoulderVel, pose._leftElbowVel]
        self._pybullet_client.resetJointStatesMultiDof(phys_model, indices,
                                                    jointPositions, jointVelocities)
        # kin_jointIndices = [
        #   0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13
        # ]
        # linkStatesKin = self._pybullet_client.getLinkStates(self._kin_model, kin_jointIndices, computeForwardKinematics=True, computeLinkVelocity=True)
        # print(linkStatesKin[2][7])
      else:
        self._pybullet_client.resetJointStateMultiDof(phys_model, chest, pose._chestRot,
                                                      pose._chestVel)
        self._pybullet_client.resetJointStateMultiDof(phys_model, neck, pose._neckRot, pose._neckVel)
        self._pybullet_client.resetJointStateMultiDof(phys_model, rightHip, pose._rightHipRot,
                                                      pose._rightHipVel)
        self._pybullet_client.resetJointStateMultiDof(phys_model, rightKnee, pose._rightKneeRot,
                                                      pose._rightKneeVel)
        self._pybullet_client.resetJointStateMultiDof(phys_model, rightAnkle, pose._rightAnkleRot,
                                                      pose._rightAnkleVel)
        self._pybullet_client.resetJointStateMultiDof(phys_model, rightShoulder,
                                                      pose._rightShoulderRot, pose._rightShoulderVel)
        self._pybullet_client.resetJointStateMultiDof(phys_model, rightElbow, pose._rightElbowRot,
                                                      pose._rightElbowVel)
        self._pybullet_client.resetJointStateMultiDof(phys_model, leftHip, pose._leftHipRot,
                                                      pose._leftHipVel)
        self._pybullet_client.resetJointStateMultiDof(phys_model, leftKnee, pose._leftKneeRot,
                                                      pose._leftKneeVel)
        self._pybullet_client.resetJointStateMultiDof(phys_model, leftAnkle, pose._leftAnkleRot,
                                                      pose._leftAnkleVel)
        self._pybullet_client.resetJointStateMultiDof(phys_model, leftShoulder,
                                                      pose._leftShoulderRot, pose._leftShoulderVel)
        self._pybullet_client.resetJointStateMultiDof(phys_model, leftElbow, pose._leftElbowRot,
                                                      pose._leftElbowVel)
    else:
      
      if initBase:
        self._pybullet_client.resetBasePositionAndOrientation(phys_model, pose._basePos,
                                                              pose._baseOrn)
      if useArray:
        indices = [chest,neck,rightHip,rightKnee,
                  rightAnkle, rightShoulder, rightElbow,leftHip,
                  leftKnee, leftAnkle, leftShoulder,leftElbow]
        jointPositions = [pose._chestRot, pose._neckRot, pose._rightHipRot, pose._rightKneeRot,
                          pose._rightAnkleRot, pose._rightShoulderRot, pose._rightElbowRot, pose._leftHipRot,
                          pose._leftKneeRot, pose._leftAnkleRot, pose._leftShoulderRot, pose._leftElbowRot]
        self._pybullet_client.resetJointStatesMultiDof(phys_model, indices,jointPositions)
        
      else:
        self._pybullet_client.resetJointStateMultiDof(phys_model, chest, pose._chestRot, [0, 0, 0])
        self._pybullet_client.resetJointStateMultiDof(phys_model, neck, pose._neckRot, [0, 0, 0])
        self._pybullet_client.resetJointStateMultiDof(phys_model, rightHip, pose._rightHipRot,
                                                      [0, 0, 0])
        self._pybullet_client.resetJointStateMultiDof(phys_model, rightKnee, pose._rightKneeRot, [0])
        self._pybullet_client.resetJointStateMultiDof(phys_model, rightAnkle, pose._rightAnkleRot,
                                                      [0, 0, 0])
        self._pybullet_client.resetJointStateMultiDof(phys_model, rightShoulder,
                                                      pose._rightShoulderRot, [0, 0, 0])
        self._pybullet_client.resetJointStateMultiDof(phys_model, rightElbow, pose._rightElbowRot,
                                                      [0])
        self._pybullet_client.resetJointStateMultiDof(phys_model, leftHip, pose._leftHipRot,
                                                      [0, 0, 0])
        self._pybullet_client.resetJointStateMultiDof(phys_model, leftKnee, pose._leftKneeRot, [0])
        self._pybullet_client.resetJointStateMultiDof(phys_model, leftAnkle, pose._leftAnkleRot,
                                                      [0, 0, 0])
        self._pybullet_client.resetJointStateMultiDof(phys_model, leftShoulder,
                                                      pose._leftShoulderRot, [0, 0, 0])
        self._pybullet_client.resetJointStateMultiDof(phys_model, leftElbow, pose._leftElbowRot, [0])

  def calcCycleCount(self, simTime, cycleTime):
    phases = simTime / cycleTime
    count = math.floor(phases)
    loop = True
    #count = (loop) ? count : cMathUtil::Clamp(count, 0, 1);
    return count

  def getCycleTime(self):
    keyFrameDuration = self._mocap_data.KeyFrameDuraction()
    cycleTime = keyFrameDuration * (self._mocap_data.NumFrames() - 1)
    return cycleTime

  def setSimTime(self, t):
    self._simTime = t
    #print("SetTimeTime time =",t)
    keyFrameDuration = self._mocap_data.KeyFrameDuraction()
    cycleTime = self.getCycleTime()
    #print("self._motion_data.NumFrames()=",self._mocap_data.NumFrames())
    self._cycleCount = self.calcCycleCount(t, cycleTime)
    #print("cycles=",cycles)
    frameTime = t - self._cycleCount * cycleTime
    if (frameTime < 0):
      frameTime += cycleTime

    #print("keyFrameDuration=",keyFrameDuration)
    #print("frameTime=",frameTime)
    self._frame = int(frameTime / keyFrameDuration)
    #print("self._frame=",self._frame)

    self._frameNext = self._frame + 1
    if (self._frameNext >= self._mocap_data.NumFrames()):
      self._frameNext = self._frame

    self._frameFraction = (frameTime - self._frame * keyFrameDuration) / (keyFrameDuration)

  def computeCycleOffset(self):
    firstFrame = 0
    lastFrame = self._mocap_data.NumFrames() - 1
    frameData = self._mocap_data._motion_data['Frames'][0]
    frameDataNext = self._mocap_data._motion_data['Frames'][lastFrame]

    basePosStart = [frameData[1], frameData[2], frameData[3]]
    basePosEnd = [frameDataNext[1], frameDataNext[2], frameDataNext[3]]
    self._cycleOffset = [
        basePosEnd[0] - basePosStart[0], basePosEnd[1] - basePosStart[1],
        basePosEnd[2] - basePosStart[2]
    ]
    return self._cycleOffset

  def computePose(self, frameFraction):
    frameData = self._mocap_data._motion_data['Frames'][self._frame]
    frameDataNext = self._mocap_data._motion_data['Frames'][self._frameNext]

    self._poseInterpolator.Slerp(frameFraction, frameData, frameDataNext, self._pybullet_client)
    #print("self._poseInterpolator.Slerp(", frameFraction,")=", pose)
    self.computeCycleOffset()
    oldPos = self._poseInterpolator._basePos
    self._poseInterpolator._basePos = [
        oldPos[0] + self._cycleCount * self._cycleOffset[0],
        oldPos[1] + self._cycleCount * self._cycleOffset[1],
        oldPos[2] + self._cycleCount * self._cycleOffset[2]
    ]
    pose = self._poseInterpolator.GetPose()

    return pose

  def convertActionToPose(self, action):
    pose = self._poseInterpolator.ConvertFromAction(self._pybullet_client, action)
    return pose

  def computeAndApplyPDForces(self, desiredPositions, maxForces):
    dofIndex = 7
    scaling = 1
    indices = []
    forces = []
    targetPositions=[]
    targetVelocities=[]
    kps = []
    kds = []
    
    for index in range(len(self._jointIndicesAll)):
      jointIndex = self._jointIndicesAll[index]
      indices.append(jointIndex)
      kps.append(self._kpOrg[dofIndex])
      kds.append(self._kdOrg[dofIndex])
      if self._jointDofCounts[index] == 4:
        force = [
            scaling * maxForces[dofIndex + 0],
            scaling * maxForces[dofIndex + 1],
            scaling * maxForces[dofIndex + 2]
        ]
        targetVelocity = [0,0,0]
        targetPosition = [
            desiredPositions[dofIndex + 0],
            desiredPositions[dofIndex + 1],
            desiredPositions[dofIndex + 2],
            desiredPositions[dofIndex + 3]
        ]
      if self._jointDofCounts[index] == 1:
        force = [scaling * maxForces[dofIndex]]
        targetPosition = [desiredPositions[dofIndex + 0]]
        targetVelocity = [0]
      forces.append(force)
      targetPositions.append(targetPosition)
      targetVelocities.append(targetVelocity)
      dofIndex += self._jointDofCounts[index]
      
    #static char* kwlist[] = { "bodyUniqueId", 
    #"jointIndices", 
    #"controlMode", "targetPositions", "targetVelocities", "forces", "positionGains", "velocityGains", "maxVelocities", "physicsClientId", NULL };
    self._pybullet_client.setJointMotorControlMultiDofArray(self._sim_model,
                                                           indices,
                                                           self._pybullet_client.STABLE_PD_CONTROL,
                                                           targetPositions = targetPositions,
                                                           targetVelocities = targetVelocities,
                                                           forces=forces,
                                                           positionGains = kps,
                                                           velocityGains = kds,
                                                           )

  def computePDForces(self, desiredPositions, desiredVelocities, maxForces):
    """Compute torques from the PD controller."""
    if desiredVelocities == None:
      desiredVelocities = [0] * self._totalDofs

    taus = self._stablePD.computePD(bodyUniqueId=self._sim_model,
                                    jointIndices=self._jointIndicesAll,
                                    desiredPositions=desiredPositions,
                                    desiredVelocities=desiredVelocities,
                                    kps=self._kpOrg,
                                    kds=self._kdOrg,
                                    maxForces=maxForces,
                                    timeStep=self._timeStep)
    return taus

  def applyPDForces(self, taus):
    """Apply pre-computed torques."""
    dofIndex = 7
    scaling = 1
    useArray = True
    indices = []
    forces = []
    
    if (useArray):
      for index in range(len(self._jointIndicesAll)):
        jointIndex = self._jointIndicesAll[index]
        indices.append(jointIndex)
        if self._jointDofCounts[index] == 4:
          force = [
              scaling * taus[dofIndex + 0], scaling * taus[dofIndex + 1],
              scaling * taus[dofIndex + 2]
          ]
        if self._jointDofCounts[index] == 1:
          force = [scaling * taus[dofIndex]]
          #print("force[", jointIndex,"]=",force)
        forces.append(force)
        dofIndex += self._jointDofCounts[index]
      self._pybullet_client.setJointMotorControlMultiDofArray(self._sim_model,
                                                             indices,
                                                             self._pybullet_client.TORQUE_CONTROL,
                                                             forces=forces)
    else:
      for index in range(len(self._jointIndicesAll)):
        jointIndex = self._jointIndicesAll[index]
        if self._jointDofCounts[index] == 4:
          force = [
              scaling * taus[dofIndex + 0], scaling * taus[dofIndex + 1],
              scaling * taus[dofIndex + 2]
          ]
          #print("force[", jointIndex,"]=",force)
          self._pybullet_client.setJointMotorControlMultiDof(self._sim_model,
                                                             jointIndex,
                                                             self._pybullet_client.TORQUE_CONTROL,
                                                             force=force)
        if self._jointDofCounts[index] == 1:
          force = [scaling * taus[dofIndex]]
          #print("force[", jointIndex,"]=",force)
          self._pybullet_client.setJointMotorControlMultiDof(
              self._sim_model,
              jointIndex,
              controlMode=self._pybullet_client.TORQUE_CONTROL,
              force=force)
        dofIndex += self._jointDofCounts[index]

  def setJointMotors(self, desiredPositions, maxForces):
    controlMode = self._pybullet_client.POSITION_CONTROL
    startIndex = 7
    chest = 1
    neck = 2
    rightHip = 3
    rightKnee = 4
    rightAnkle = 5
    rightShoulder = 6
    rightElbow = 7
    leftHip = 9
    leftKnee = 10
    leftAnkle = 11
    leftShoulder = 12
    leftElbow = 13
    kp = 0.2

    forceScale = 1
    #self._jointDofCounts=[4,4,4,1,4,4,1,4,1,4,4,1]
    maxForce = [
        forceScale * maxForces[startIndex], forceScale * maxForces[startIndex + 1],
        forceScale * maxForces[startIndex + 2], forceScale * maxForces[startIndex + 3]
    ]
    startIndex += 4
    self._pybullet_client.setJointMotorControlMultiDof(
        self._sim_model,
        chest,
        controlMode,
        targetPosition=self._poseInterpolator._chestRot,
        positionGain=kp,
        force=maxForce)
    maxForce = [
        maxForces[startIndex], maxForces[startIndex + 1], maxForces[startIndex + 2],
        maxForces[startIndex + 3]
    ]
    startIndex += 4
    self._pybullet_client.setJointMotorControlMultiDof(
        self._sim_model,
        neck,
        controlMode,
        targetPosition=self._poseInterpolator._neckRot,
        positionGain=kp,
        force=maxForce)
    maxForce = [
        maxForces[startIndex], maxForces[startIndex + 1], maxForces[startIndex + 2],
        maxForces[startIndex + 3]
    ]
    startIndex += 4
    self._pybullet_client.setJointMotorControlMultiDof(
        self._sim_model,
        rightHip,
        controlMode,
        targetPosition=self._poseInterpolator._rightHipRot,
        positionGain=kp,
        force=maxForce)
    maxForce = [forceScale * maxForces[startIndex]]
    startIndex += 1
    self._pybullet_client.setJointMotorControlMultiDof(
        self._sim_model,
        rightKnee,
        controlMode,
        targetPosition=self._poseInterpolator._rightKneeRot,
        positionGain=kp,
        force=maxForce)
    maxForce = [
        maxForces[startIndex], maxForces[startIndex + 1], maxForces[startIndex + 2],
        maxForces[startIndex + 3]
    ]
    startIndex += 4
    self._pybullet_client.setJointMotorControlMultiDof(
        self._sim_model,
        rightAnkle,
        controlMode,
        targetPosition=self._poseInterpolator._rightAnkleRot,
        positionGain=kp,
        force=maxForce)
    maxForce = [
        forceScale * maxForces[startIndex], forceScale * maxForces[startIndex + 1],
        forceScale * maxForces[startIndex + 2], forceScale * maxForces[startIndex + 3]
    ]
    startIndex += 4
    maxForce = [forceScale * maxForces[startIndex]]
    startIndex += 1
    self._pybullet_client.setJointMotorControlMultiDof(
        self._sim_model,
        rightElbow,
        controlMode,
        targetPosition=self._poseInterpolator._rightElbowRot,
        positionGain=kp,
        force=maxForce)
    maxForce = [
        maxForces[startIndex], maxForces[startIndex + 1], maxForces[startIndex + 2],
        maxForces[startIndex + 3]
    ]
    startIndex += 4
    self._pybullet_client.setJointMotorControlMultiDof(
        self._sim_model,
        leftHip,
        controlMode,
        targetPosition=self._poseInterpolator._leftHipRot,
        positionGain=kp,
        force=maxForce)
    maxForce = [forceScale * maxForces[startIndex]]
    startIndex += 1
    self._pybullet_client.setJointMotorControlMultiDof(
        self._sim_model,
        leftKnee,
        controlMode,
        targetPosition=self._poseInterpolator._leftKneeRot,
        positionGain=kp,
        force=maxForce)
    maxForce = [
        maxForces[startIndex], maxForces[startIndex + 1], maxForces[startIndex + 2],
        maxForces[startIndex + 3]
    ]
    startIndex += 4
    self._pybullet_client.setJointMotorControlMultiDof(
        self._sim_model,
        leftAnkle,
        controlMode,
        targetPosition=self._poseInterpolator._leftAnkleRot,
        positionGain=kp,
        force=maxForce)
    maxForce = [
        maxForces[startIndex], maxForces[startIndex + 1], maxForces[startIndex + 2],
        maxForces[startIndex + 3]
    ]
    startIndex += 4
    self._pybullet_client.setJointMotorControlMultiDof(
        self._sim_model,
        leftShoulder,
        controlMode,
        targetPosition=self._poseInterpolator._leftShoulderRot,
        positionGain=kp,
        force=maxForce)
    maxForce = [forceScale * maxForces[startIndex]]
    startIndex += 1
    self._pybullet_client.setJointMotorControlMultiDof(
        self._sim_model,
        leftElbow,
        controlMode,
        targetPosition=self._poseInterpolator._leftElbowRot,
        positionGain=kp,
        force=maxForce)
    #print("startIndex=",startIndex)

  def getPhase(self):
    keyFrameDuration = self._mocap_data.KeyFrameDuraction()
    cycleTime = keyFrameDuration * (self._mocap_data.NumFrames() - 1)
    phase = self._simTime / cycleTime
    phase = math.fmod(phase, 1.0)
    if (phase < 0):
      phase += 1
    return phase

  def buildHeadingTrans(self, rootOrn):
    #align root transform 'forward' with world-space x axis
    eul = self._pybullet_client.getEulerFromQuaternion(rootOrn)
    refDir = [1, 0, 0]
    rotVec = self._pybullet_client.rotateVector(rootOrn, refDir)
    heading = math.atan2(-rotVec[2], rotVec[0])
    heading2 = eul[1]
    #print("heading=",heading)
    headingOrn = self._pybullet_client.getQuaternionFromAxisAngle([0, 1, 0], -heading)
    return headingOrn

  def buildOriginTrans(self):
    rootPos, rootOrn = self._pybullet_client.getBasePositionAndOrientation(self._sim_model)

    #print("rootPos=",rootPos, " rootOrn=",rootOrn)
    invRootPos = [-rootPos[0], 0, -rootPos[2]]
    #invOrigTransPos, invOrigTransOrn = self._pybullet_client.invertTransform(rootPos,rootOrn)
    headingOrn = self.buildHeadingTrans(rootOrn)
    #print("headingOrn=",headingOrn)
    headingMat = self._pybullet_client.getMatrixFromQuaternion(headingOrn)
    #print("headingMat=",headingMat)
    #dummy, rootOrnWithoutHeading = self._pybullet_client.multiplyTransforms([0,0,0],headingOrn, [0,0,0], rootOrn)
    #dummy, invOrigTransOrn = self._pybullet_client.multiplyTransforms([0,0,0],rootOrnWithoutHeading, invOrigTransPos, invOrigTransOrn)

    invOrigTransPos, invOrigTransOrn = self._pybullet_client.multiplyTransforms([0, 0, 0],
                                                                                headingOrn,
                                                                                invRootPos,
                                                                                [0, 0, 0, 1])
    #print("invOrigTransPos=",invOrigTransPos)
    #print("invOrigTransOrn=",invOrigTransOrn)
    invOrigTransMat = self._pybullet_client.getMatrixFromQuaternion(invOrigTransOrn)
    #print("invOrigTransMat =",invOrigTransMat )
    return invOrigTransPos, invOrigTransOrn

  def getState(self):
    stateVector = []
    phase = self.getPhase()
    #print("phase=",phase)
    stateVector.append(phase)

    rootTransPos, rootTransOrn = self.buildOriginTrans()
    basePos, baseOrn = self._pybullet_client.getBasePositionAndOrientation(self._sim_model)

    rootPosRel, dummy = self._pybullet_client.multiplyTransforms(rootTransPos, rootTransOrn,
                                                                 basePos, [0, 0, 0, 1])
    #print("!!!rootPosRel =",rootPosRel )
    #print("rootTransPos=",rootTransPos)
    #print("basePos=",basePos)
    localPos, localOrn = self._pybullet_client.multiplyTransforms(rootTransPos, rootTransOrn,
                                                                  basePos, baseOrn)

    localPos = [
        localPos[0] - rootPosRel[0], localPos[1] - rootPosRel[1], localPos[2] - rootPosRel[2]
    ]
    #print("localPos=",localPos)

    stateVector.append(rootPosRel[1])

    #self.pb2dmJoints=[0,1,2,9,10,11,3,4,5,12,13,14,6,7,8]
    # self.pb2dmJoints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    linkIndicesSim = [
      back_bkz, back_bkx,
      l_arm_ely, l_arm_elx, l_arm_wry2,
      neck_ry,
      r_arm_ely, r_arm_elx, r_arm_wry2,
      l_leg_hpy, l_leg_kny, l_leg_akx,
      r_leg_hpy, r_leg_kny, r_leg_akx
    ]

    # linkIndicesSim = []
    # for pbJoint in range(self._pybullet_client.getNumJoints(self._sim_model)):
    #   linkIndicesSim.append(self.pb2dmJoints[pbJoint])
      
    linkStatesSim = self._pybullet_client.getLinkStates(self._sim_model, linkIndicesSim, computeForwardKinematics=True, computeLinkVelocity=True)

    for index in range(len(linkIndicesSim)):
      # j = self.pb2dmJoints[pbJoint]
      #print("joint order:",j)
      #ls = self._pybullet_client.getLinkState(self._sim_model, j, computeForwardKinematics=True)
      ls = linkStatesSim[index]
      linkPos = ls[0]
      linkOrn = ls[1]
      linkPosLocal, linkOrnLocal = self._pybullet_client.multiplyTransforms(rootTransPos, rootTransOrn, linkPos, linkOrn)
      
      if (linkOrnLocal[3] < 0):
       linkOrnLocal = [-linkOrnLocal[0], -linkOrnLocal[1], -linkOrnLocal[2], -linkOrnLocal[3]]
      
      linkPosLocal = [
          linkPosLocal[0] - rootPosRel[0], linkPosLocal[1] - rootPosRel[1],
          linkPosLocal[2] - rootPosRel[2]
      ]

      for l in linkPosLocal:
        stateVector.append(l)
      #re-order the quaternion, DeepMimic uses w,x,y,z

      stateVector.append(linkOrnLocal[3])
      stateVector.append(linkOrnLocal[0])
      stateVector.append(linkOrnLocal[1])
      stateVector.append(linkOrnLocal[2])
       
    for index in range(len(linkIndicesSim)):
      # j = self.pb2dmJoints[pbJoint]
      #ls = self._pybullet_client.getLinkState(self._sim_model, j, computeLinkVelocity=True)
      ls = linkStatesSim[index]
      
      linkLinVel = ls[6]
      linkAngVel = ls[7]
      linkLinVelLocal, unused = self._pybullet_client.multiplyTransforms([0, 0, 0], rootTransOrn,
                                                                         linkLinVel, [0, 0, 0, 1])
      #linkLinVelLocal=[linkLinVelLocal[0]-rootPosRel[0],linkLinVelLocal[1]-rootPosRel[1],linkLinVelLocal[2]-rootPosRel[2]]
      linkAngVelLocal, unused = self._pybullet_client.multiplyTransforms([0, 0, 0], rootTransOrn,
                                                                         linkAngVel, [0, 0, 0, 1])

      for l in linkLinVelLocal:
        stateVector.append(l)
      for l in linkAngVelLocal:
        stateVector.append(l)

    #print("stateVector len=",len(stateVector))
    #for st in range (len(stateVector)):
    #  print("state[",st,"]=",stateVector[st])
    return stateVector

  def terminates(self):
    #check if any non-allowed body part hits the ground
    terminates = False
    pts = self._pybullet_client.getContactPoints()
    for p in pts:
      part = -1
      #ignore self-collision
      if (p[1] == p[2]):
        continue
      if (p[1] == self._sim_model):
        part = p[3]
      if (p[2] == self._sim_model):
        part = p[4]
      if (part >= 0 and part in self._fall_contact_body_parts):
        #print("terminating part:", part)
        terminates = True

    return terminates

  def quatMul(self, q1, q2):
    return [
        q1[3] * q2[0] + q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1],
        q1[3] * q2[1] + q1[1] * q2[3] + q1[2] * q2[0] - q1[0] * q2[2],
        q1[3] * q2[2] + q1[2] * q2[3] + q1[0] * q2[1] - q1[1] * q2[0],
        q1[3] * q2[3] - q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2]
    ]

  def calcRootAngVelErr(self, vel0, vel1):
    diff = [vel0[0] - vel1[0], vel0[1] - vel1[1], vel0[2] - vel1[2]]
    return diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]

  def calcRootRotDiff(self, orn0, orn1):
    orn0Conj = [-orn0[0], -orn0[1], -orn0[2], orn0[3]]
    q_diff = self.quatMul(orn1, orn0Conj)
    axis, angle = self._pybullet_client.getAxisAngleFromQuaternion(q_diff)
    return angle * angle

  def getReward(self, pose):
    """Compute and return the pose-based reward."""
    # from DeepMimic double cSceneImitate::CalcRewardImitate
    # todo: compensate for ground height in some parts, once we move to non-flat terrain
    # not values from the paper, but from the published code.

    useRootReward = OFF
    pose_w = 0.5
    # pose_w = 0.65       # from paper
    vel_w = 0.05
    # vel_w = 0.1         # from paper
    end_eff_w = 0.15
    # end_eff_w = 0.3   # sub
    # does not exist in paper
    if self._useComReward:
      com_w = 0.1
    else:
      com_w = 0
    
    root_rot_w = 0.20833
    if useRootReward == OFF:
      root_w = 0
    else:
      root_w = 0.2
      # root_w = 0.4      # sub

    total_w = pose_w + vel_w + end_eff_w + root_w + com_w
    pose_w /= total_w
    vel_w /= total_w
    end_eff_w /= total_w
    root_w /= total_w
    com_w /= total_w

    pose_scale = 2
    vel_scale = 0.1
    end_eff_scale = 40
    root_scale = 5
    com_scale = 10
    err_scale = 1  # error scale

    reward = 0

    pose_err = 0
    vel_err = 0
    end_eff_err = 0
    root_err = 0
    com_err = 0
    heading_err = 0

    if self._useComReward:
      comSim, comSimVel = self.computeCOMposVel(self._sim_model)
      comKin, comKinVel = self.computeCOMposVel(self._kin_model)
    
    # jointIndex = [
    #   back_bkx, neck_ry,
    #   r_leg_hpy, r_leg_kny, r_leg_akx, r_arm_ely, r_arm_elx,
    #   l_leg_hpy, l_leg_kny, l_leg_akx, l_arm_ely, l_arm_elx
    # ]

    # mJointWeights = [
    #     0.10416, 0.0625, 0.10416, 0.0625, 0.041666666666666671, 0.0625, 0.0416,
    #     0.10416, 0.0625, 0.0416, 0.0625, 0.0416
    # ]

    mJointWeights = [
        0.10416, 0.10416, 0.10416,
        0.0625,
        0.10416, 0.10416, 0.10416,
        0.0625,
        0.0416, 0.0416,
        0.0625, 0.0625, 0.0625,
        0.0416,
        0.10416, 0.10416, 0.10416,
        0.0625,
        0.0416, 0.0416,
        0.0625, 0.0625, 0.0625,
        0.0416
    ]

    # chest * 2 hip * 2
    # mJointWeights = [
    #     0.20832, 0.20832, 0.20832,
    #     0.0625,
    #     0.20832, 0.20832, 0.20832,
    #     0.0625,
    #     0.0416, 0.0416,
    #     0.0625, 0.0625, 0.0625,
    #     0.0416,
    #     0.20832, 0.20832, 0.20832,
    #     0.0625,
    #     0.0416, 0.0416,
    #     0.0625, 0.0625, 0.0625,
    #     0.0416
    # ]

    # mJointWeights = [
    #     0.333, 0.333, 0.333,
    #     1.0,
    #     0.333, 0.333, 0.333,
    #     1.0,
    #     0.5, 0.5,
    #     0.333, 0.333, 0.333,
    #     0.333,
    #     0.333, 0.333, 0.333,
    #     1.0,
    #     0.5, 0.5,
    #     0.333, 0.333, 0.333,
    #     1.0
    # ]

    # all one
    # mJointWeights = [
    #     1.0, 1.0, 1.0,
    #     1.0,
    #     1.0, 1.0, 1.0,
    #     1.0,
    #     1.0, 1.0,
    #     1.0, 1.0, 1.0,
    #     1.0,
    #     1.0, 1.0, 1.0,
    #     1.0,
    #     1.0, 1.0,
    #     1.0, 1.0, 1.0,
    #     1.0
    # ]

    jointIndices = [
      back_bkz, back_bky, back_bkx,
      neck_ry,
      r_leg_hpz, r_leg_hpx, r_leg_hpy,
      r_leg_kny,
      r_leg_aky, r_leg_akx,
      r_arm_shz, r_arm_shx, r_arm_ely,
      r_arm_elx,
      l_leg_hpz, l_leg_hpx, l_leg_hpy,
      l_leg_kny,
      l_leg_aky, l_leg_akx,
      l_arm_shz, l_arm_shx, l_arm_ely,
      l_arm_elx
    ]
    
    jointIndex = [
      back_bkx, neck_ry,
      r_leg_hpy, r_leg_kny, r_leg_akx, r_arm_ely, r_arm_elx,
      l_leg_hpy, l_leg_kny, l_leg_akx, l_arm_ely, l_arm_elx
    ]

    kin_jointIndices = [
      1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13
    ]

    num_end_effs = 0
    # num_joints = 15

    if self._log_mode == ON:
      log_err_data = []
      log_sim_data = []
      log_kin_data = []

    rootPosSim, rootOrnSim = self._pybullet_client.getBasePositionAndOrientation(self._sim_model)
    rootPosKin, rootOrnKin = self._pybullet_client.getBasePositionAndOrientation(self._kin_model)
    linVelSim, angVelSim = self._pybullet_client.getBaseVelocity(self._sim_model)
    #don't read the velocities from the kinematic model (they are zero), use the pose interpolator velocity
    #see also issue https://github.com/bulletphysics/bullet3/issues/2401
    linVelKin = self._poseInterpolator._baseLinVel
    angVelKin = self._poseInterpolator._baseAngVel
    
    root_orn_sim = self.quatMul(rootOrnSim, self._y2z)
    root_rot_err = self.calcRootRotDiff(root_orn_sim, rootOrnKin)
    pose_err += root_rot_w * root_rot_err

    root_pos_diff = [
        rootPosSim[0] - rootPosKin[0], rootPosSim[1] - rootPosKin[1], rootPosSim[2] - rootPosKin[2]
    ]
    root_pos_err = root_pos_diff[0] * root_pos_diff[0] + root_pos_diff[1] * root_pos_diff[
        1] + root_pos_diff[2] * root_pos_diff[2]

    root_vel_diff = [
        linVelSim[0] - linVelKin[0], linVelSim[2] - linVelKin[1], -linVelSim[1] - linVelKin[2]
    ]
    # root_vel_diff = [
    #     linVelSim[0] - linVelKin[0], linVelSim[1] - linVelKin[1], linVelSim[2] - linVelKin[2]
    # ]
    root_vel_err = root_vel_diff[0] * root_vel_diff[0] + root_vel_diff[1] * root_vel_diff[
        1] + root_vel_diff[2] * root_vel_diff[2]

    root_ang_vel_err = self.calcRootAngVelErr(angVelSim, angVelKin)
    vel_err += root_rot_w * root_ang_vel_err

    # log about root
    if self._log_mode == ON:
      #log err
      orn0Conj = [-root_orn_sim[0], -root_orn_sim[1], -root_orn_sim[2], root_orn_sim[3]]
      q_diff = self.quatMul(rootOrnKin, orn0Conj)
      log_err_data.append(root_pos_diff[0])
      log_err_data.append(root_pos_diff[1])
      log_err_data.append(root_pos_diff[2])
      log_err_data.append(root_vel_diff[0])
      log_err_data.append(root_vel_diff[1])
      log_err_data.append(root_vel_diff[2])
      axis, angle = self._pybullet_client.getAxisAngleFromQuaternion(q_diff)
      log_err_data.append(angle*180/math.pi)
      rpy = self._pybullet_client.getEulerFromQuaternion(q_diff)
      log_err_data.append(rpy[0]*180/math.pi)
      log_err_data.append(rpy[1]*180/math.pi)
      log_err_data.append(rpy[2]*180/math.pi)
      root_ang_vel_diff = [angVelSim[0] - angVelKin[0], angVelSim[1] - angVelKin[1], angVelSim[2] - angVelKin[2]]
      log_err_data.append(root_ang_vel_diff[0]*180/math.pi)
      log_err_data.append(root_ang_vel_diff[1]*180/math.pi)
      log_err_data.append(root_ang_vel_diff[2]*180/math.pi)
      #log_sim
      log_sim_data.append(rootPosSim[0])
      log_sim_data.append(rootPosSim[1])
      log_sim_data.append(rootPosSim[2])
      log_sim_data.append(linVelSim[0])
      log_sim_data.append(linVelSim[1])
      log_sim_data.append(linVelSim[2])
      axis, angle = self._pybullet_client.getAxisAngleFromQuaternion(root_orn_sim)
      log_sim_data.append(angle*180/math.pi)
      rpy = self._pybullet_client.getEulerFromQuaternion(root_orn_sim)
      log_sim_data.append(rpy[0]*180/math.pi)
      log_sim_data.append(rpy[1]*180/math.pi)
      log_sim_data.append(rpy[2]*180/math.pi)
      log_sim_data.append(angVelSim[0]*180/math.pi)
      log_sim_data.append(angVelSim[1]*180/math.pi)
      log_sim_data.append(angVelSim[2]*180/math.pi)
      #log_kin
      log_kin_data.append(rootPosKin[0])
      log_kin_data.append(rootPosKin[1])
      log_kin_data.append(rootPosKin[2])
      log_kin_data.append(linVelKin[0])
      log_kin_data.append(linVelKin[1])
      log_kin_data.append(linVelKin[2])
      axis, angle = self._pybullet_client.getAxisAngleFromQuaternion(rootOrnKin)
      log_kin_data.append(angle*180/math.pi)
      rpy = self._pybullet_client.getEulerFromQuaternion(rootOrnKin)
      log_kin_data.append(rpy[0]*180/math.pi)
      log_kin_data.append(rpy[1]*180/math.pi)
      log_kin_data.append(rpy[2]*180/math.pi)
      log_kin_data.append(angVelKin[0]*180/math.pi)
      log_kin_data.append(angVelKin[1]*180/math.pi)
      log_kin_data.append(angVelKin[2]*180/math.pi)

    # log about COM
    if self._log_mode == ON:
      comSim, comSimVel = self.computeCOMposVel(self._sim_model)
      comKin, comKinVel = self.computeCOMposVel(self._kin_model)
      # log_err
      err_com_pos = comSim - comKin
      err_com_vel = comSimVel - comKinVel
      log_err_data.append(err_com_pos[0])
      log_err_data.append(err_com_pos[1])
      log_err_data.append(err_com_pos[2])
      log_err_data.append(err_com_vel[0])
      log_err_data.append(err_com_vel[1])
      log_err_data.append(err_com_vel[2])
      # log_sim
      log_sim_data.append(comSim[0])
      log_sim_data.append(comSim[1])
      log_sim_data.append(comSim[2])
      log_sim_data.append(comSimVel[0])
      log_sim_data.append(comSimVel[1])
      log_sim_data.append(comSimVel[2])
      # log_kin
      log_kin_data.append(comKin[0])
      log_kin_data.append(comKin[1])
      log_kin_data.append(comKin[2])
      log_kin_data.append(comKinVel[0])
      log_kin_data.append(comKinVel[1])
      log_kin_data.append(comKinVel[2])

    useArray = True

    if useArray:
      simJointStates = self._pybullet_client.getJointStates(self._sim_model, jointIndices)
      kinJointStates = self._pybullet_client.getJointStates(self._kin_model, jointIndices)
    if useArray:
      linkStatesSim = self._pybullet_client.getLinkStates(self._sim_model, jointIndices, computeForwardKinematics=True, computeLinkVelocity=True)
      linkStatesKin = self._pybullet_client.getLinkStates(self._kin_model, jointIndices, computeForwardKinematics=True, computeLinkVelocity=True)
      endEffectorStatesSim = self._pybullet_client.getLinkStates(self._sim_model, self._end_effectors, computeForwardKinematics=True, computeLinkVelocity=True)
      endEffectorStatesKin = self._pybullet_client.getLinkStates(self._kin_model, self._end_effectors, computeForwardKinematics=True, computeLinkVelocity=True)

    for index in range(len(jointIndices)):
      curr_pose_err = 0
      curr_vel_err = 0
      w = mJointWeights[index]

      if useArray:
        simJointInfo = simJointStates[index]
        simLinkInfo = linkStatesSim[index]
      if useArray:
        kinJointInfo = kinJointStates[index]
        kinLinkInfo = linkStatesKin[index]
      
      angle = simJointInfo[0] - kinJointInfo[0]
      curr_pose_err = angle * angle
      velDiff = simJointInfo[1] - kinJointInfo[1]
      curr_vel_err = velDiff * velDiff
      if self._log_mode == ON:
        # log err
        log_err_data.append(angle*180/math.pi)
        log_err_data.append(velDiff*180/math.pi)
        # log_sim
        log_sim_data.append(simJointInfo[0]*180/math.pi)
        log_sim_data.append(simJointInfo[1]*180/math.pi)
        # log_kin
        log_kin_data.append(kinJointInfo[0]*180/math.pi)
        log_kin_data.append(kinJointInfo[1]*180/math.pi)
      
      pose_err += w * curr_pose_err
      vel_err += w * curr_vel_err

    # error_end_effector
    if len(self._end_effectors) != 0:
      for index in range(len(self._end_effectors)):
        simLinkInfo = endEffectorStatesSim[index]
        kinLinkInfo = endEffectorStatesKin[index]
        linkPosSim = simLinkInfo[0]
        linkPosKin = kinLinkInfo[0]
        linkPosDiff = [
            linkPosSim[0] - linkPosKin[0], linkPosSim[1] - linkPosKin[1],
            linkPosSim[2] - linkPosKin[2]
        ]
        curr_end_err = linkPosDiff[0] * linkPosDiff[0] + linkPosDiff[1] * linkPosDiff[
            1] + linkPosDiff[2] * linkPosDiff[2]
        end_eff_err += curr_end_err
        num_end_effs += 1
        if self._log_mode == ON:
          # log err
          log_err_data.append(linkPosDiff[0])
          log_err_data.append(linkPosDiff[1])
          log_err_data.append(linkPosDiff[2])
          # log sim
          log_sim_data.append(linkPosSim[0])
          log_sim_data.append(linkPosSim[1])
          log_sim_data.append(linkPosSim[2])
          # log kin
          log_kin_data.append(linkPosKin[0])
          log_kin_data.append(linkPosKin[1])
          log_kin_data.append(linkPosKin[2])

    if (num_end_effs > 0):
      end_eff_err /= num_end_effs
    
    if self._log_mode == ON:
      err_record = pd.Series(log_err_data, index=self._log_columns)
      sim_record = pd.Series(log_sim_data, index=self._log_columns)
      kin_record = pd.Series(log_kin_data, index=self._log_columns)
      self._err_log_df = self._err_log_df.append(err_record, ignore_index=True)
      self._sim_log_df = self._sim_log_df.append(sim_record, ignore_index=True)
      self._kin_log_df = self._kin_log_df.append(kin_record, ignore_index=True)
      self._err_log_df.to_csv(self._err_log_path, mode='w')
      self._sim_log_df.to_csv(self._sim_log_path, mode='w')
      self._kin_log_df.to_csv(self._kin_log_path, mode='w')

    # root_pos_diff = [
    #     rootPosSim[0] - rootPosKin[0], rootPosSim[1] - rootPosKin[1] - 0.082, rootPosSim[2] - rootPosKin[2]
    # ]
    
    #root_rot_err = cMathUtil::QuatDiffTheta(root_rot0, root_rot1)
    #root_rot_err *= root_rot_err

    #root_vel_err = (root_vel1 - root_vel0).squaredNorm()
    #root_ang_vel_err = (root_ang_vel1 - root_ang_vel0).squaredNorm()

    root_err = root_pos_err + 0.1 * root_rot_err + 0.01 * root_vel_err + 0.001 * root_ang_vel_err

    # COM error in initial code -> COM velocities
    if self._useComReward:
      # com_err = 0.1 * np.sum(np.square(comKinVel - comSimVel))
      com_err = 0.1 * np.sum(np.square(comKin - comSim))
    #com_err = 0.1 * (com_vel1_world - com_vel0_world).squaredNorm()

    #print("pose_err=",pose_err)
    #print("vel_err=",vel_err)
    pose_reward = math.exp(-err_scale * pose_scale * pose_err)
    vel_reward = math.exp(-err_scale * vel_scale * vel_err)
    end_eff_reward = math.exp(-err_scale * end_eff_scale * end_eff_err)
    # end_eff_reward = 0
    root_reward = math.exp(-err_scale * root_scale * root_err)
    com_reward = math.exp(-err_scale * com_scale * com_err)

    reward = pose_w * pose_reward + vel_w * vel_reward + end_eff_w * end_eff_reward + root_w * root_reward + com_w * com_reward

    # pose_reward,vel_reward,end_eff_reward, root_reward, com_reward);
    #print("reward=",reward)
    #print("pose_reward=",pose_reward)
    #print("vel_reward=",vel_reward)
    #print("end_eff_reward=",end_eff_reward)
    #print("root_reward=",root_reward)
    #print("com_reward=",com_reward)
    
    info_rew = dict(
      pose_reward=pose_reward,
      vel_reward=vel_reward,
      end_eff_reward=end_eff_reward,
      root_reward=root_reward,
      com_reward=com_reward
    )
    
    info_errs = dict(
      pose_err=pose_err,
      vel_err=vel_err,
      end_eff_err=end_eff_err,
      root_err=root_err,
      com_err=com_err
    )
    
    return reward

  def calcMaxReward(self):
    """Compute and return the pose-based reward."""
    #from DeepMimic double cSceneImitate::CalcRewardImitate
    #todo: compensate for ground height in some parts, once we move to non-flat terrain
    # not values from the paper, but from the published code.
    pose_w = 0.5
    vel_w = 0.05
    end_eff_w = 0.15
    # end_eff_w = 0.3   # sub
    # does not exist in paper
    root_w = 0.2
    # root_w = 0.4      # sub
    if self._useComReward:
      com_w = 0.1
    else:
      com_w = 0

    total_w = pose_w + vel_w + end_eff_w + root_w + com_w
    pose_w /= total_w
    vel_w /= total_w
    end_eff_w /= total_w
    root_w /= total_w
    com_w /= total_w

    pose_scale = 2
    vel_scale = 0.1
    end_eff_scale = 40
    root_scale = 5
    com_scale = 10
    err_scale = 1  # error scale

    reward = 0

    pose_err = 0
    vel_err = 0
    end_eff_err = 0
    root_err = 0
    com_err = 0
    heading_err = 0
    root_pos_err = 0
    root_rot_err = 0
    root_vel_err = 0
    root_ang_vel_err = 0

    root_id = 0

    root_err = root_pos_err + 0.1 * root_rot_err + 0.01 * root_vel_err + 0.001 * root_ang_vel_err

    # # COM error in initial code -> COM velocities
    # if self._useComReward:
    #   com_err = 0.1 * np.sum(np.square(comKinVel - comSimVel))
    # com_err = 0.1 * np.sum(np.square(comKin - comSim))
    #com_err = 0.1 * (com_vel1_world - com_vel0_world).squaredNorm()

    #print("pose_err=",pose_err)
    #print("vel_err=",vel_err)
    pose_reward = math.exp(-err_scale * pose_scale * pose_err)
    vel_reward = math.exp(-err_scale * vel_scale * vel_err)
    # end_eff_reward = math.exp(-err_scale * end_eff_scale * end_eff_err)
    end_eff_reward = 0
    root_reward = math.exp(-err_scale * root_scale * root_err)
    com_reward = math.exp(-err_scale * com_scale * com_err)

    reward = pose_w * pose_reward + vel_w * vel_reward + end_eff_w * end_eff_reward + root_w * root_reward + com_w * com_reward

    # pose_reward,vel_reward,end_eff_reward, root_reward, com_reward);
    #print("reward=",reward)
    #print("pose_reward=",pose_reward)
    #print("vel_reward=",vel_reward)
    #print("end_eff_reward=",end_eff_reward)
    #print("root_reward=",root_reward)
    #print("com_reward=",com_reward)
    
    info_rew = dict(
      pose_reward=pose_reward,
      vel_reward=vel_reward,
      end_eff_reward=end_eff_reward,
      root_reward=root_reward,
      com_reward=com_reward
    )
    
    info_errs = dict(
      pose_err=pose_err,
      vel_err=vel_err,
      end_eff_err=end_eff_err,
      root_err=root_err,
      com_err=com_err
    )
    
    return reward

  def computeCOMposVel(self, uid: int):
    """Compute center-of-mass position and velocity."""
    pb = self._pybullet_client
    num_joints = pb.getNumJoints(uid)
    jointIndices = range(num_joints)
    link_states = pb.getLinkStates(uid, jointIndices, computeLinkVelocity=1)
    link_pos = np.array([s[0] for s in link_states])
    link_vel = np.array([s[-2] for s in link_states])
    tot_mass = 0.
    masses = []
    for j in jointIndices:
      mass_, *_ = pb.getDynamicsInfo(uid, j)
      masses.append(mass_)
      tot_mass += mass_
    masses = np.asarray(masses)[:, None]
    com_pos = np.sum(masses * link_pos, axis=0) / tot_mass
    com_vel = np.sum(masses * link_vel, axis=0) / tot_mass
    return com_pos, com_vel
