import pybullet as p
import pybullet_data
import numpy as np
import math

from . import data as pd

def create_debug_sphere(color=np.array([0, 0, 0, 1]), radius=0.02):
  target_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
  sphere_id = p.createMultiBody(baseVisualShapeIndex=target_visual_shape,
                                basePosition=np.array([0, 0, 0]))
  return sphere_id


def load_reacher():
  p.connect(p.GUI)
  p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
  p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
  p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
  p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
  p.setAdditionalSearchPath(pybullet_data.getDataPath())
  p.resetDebugVisualizerCamera(cameraDistance=0.5,
                               cameraYaw=46,
                               cameraPitch=-30,
                               cameraTargetPosition=[0, 0, 0.1])

  URDF_PATH = pd.getDataPath() + "/pupper_arm.urdf"
  return p.loadURDF(URDF_PATH, useFixedBase=True)


def get_joint_ids(reacher_id):
  joint_ids = []
  for j in range(p.getNumJoints(reacher_id)):
    info = p.getJointInfo(reacher_id, j)
    joint_type = info[2]
    if (joint_type == p.JOINT_PRISMATIC or joint_type == p.JOINT_REVOLUTE):
      joint_ids.append(j)
  return joint_ids


def get_param_ids(reacher_id, ik: bool = False):
  param_ids = []

  if ik:
    axes = [' x', ' y', ' z']
    vals = [0.0, 0.1, 0.1]
    for i in range(len(axes)):
        param_ids.append(p.addUserDebugParameter(axes[i], -.2, .2, vals[i]))

  else:
    for j in range(p.getNumJoints(reacher_id)):
      info = p.getJointInfo(reacher_id, j)
      joint_name = info[1]
      joint_type = info[2]
      if (joint_type == p.JOINT_PRISMATIC or joint_type == p.JOINT_REVOLUTE):
        param_ids.append(
            p.addUserDebugParameter(joint_name.decode("utf-8"), -math.pi/2, math.pi/2,
                                    0))

  return param_ids


def zero_damping(reacher_id):
  p.changeDynamics(reacher_id, -1, linearDamping=0, angularDamping=0)
  for j in range(p.getNumJoints(reacher_id)):
    p.changeDynamics(reacher_id, j, linearDamping=0, angularDamping=0)
