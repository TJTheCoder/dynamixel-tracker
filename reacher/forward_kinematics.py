import math
import numpy as np
import copy

HIP_OFFSET = 0.0335
UPPER_LEG_OFFSET = 0.10 # length of link 1
LOWER_LEG_OFFSET = 0.13 # length of link 2

def rotation_matrix(axis, angle):
  """
  Create a 3x3 rotation matrix which rotates about a specific axis

  Args:
    axis:  Array.  Unit vector in the direction of the axis of rotation
    angle: Number. The amount to rotate about the axis in radians

  Returns:
    3x3 rotation matrix as a numpy array
  """

  axis = np.asarray(axis, dtype=np.float64).reshape(3)
  axis_norm = np.linalg.norm(axis)
  if axis_norm == 0:
    return np.eye(3)
  axis = axis / axis_norm

  x, y, z = axis
  skew = np.array([
    [0, -z, y],
    [z, 0, -x],
    [-y, x, 0],
  ])

  rot_mat = (
    np.eye(3)
    + math.sin(angle) * skew
    + (1.0 - math.cos(angle)) * (skew @ skew)
  )
  return rot_mat

def homogenous_transformation_matrix(axis, angle, v_A):
  """
  Create a 4x4 transformation matrix which transforms from frame A to frame B

  Args:
    axis:  Array.  Unit vector in the direction of the axis of rotation
    angle: Number. The amount to rotate about the axis in radians
    v_A:   Vector. The vector translation from A to B defined in frame A

  Returns:
    4x4 transformation matrix as a numpy array
  """

  v_A = np.asarray(v_A, dtype=np.float64).reshape(3, 1)
  T = np.block([
    [rotation_matrix(axis, angle), v_A],
    [np.zeros((1, 3)), np.ones((1, 1))],
  ])
  return T

def fk_hip(joint_angles):
  """
  Use forward kinematics equations to calculate the xyz coordinates of the hip
  frame given the joint angles of the robot

  Args:
    joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle, 
                  elbow_angle]. Angles are in radians
  Returns:
    4x4 matrix representing the pose of the hip frame in the base frame
  """

  hip_angle = np.asarray(joint_angles, dtype=np.float64).reshape(3)[0]
  hip_frame = homogenous_transformation_matrix(
    axis=np.array([0.0, 0.0, 1.0]),
    angle=hip_angle,
    v_A=np.zeros(3),
  )
  return hip_frame

def fk_shoulder(joint_angles):
  """
  Use forward kinematics equations to calculate the xyz coordinates of the shoulder
  joint given the joint angles of the robot

  Args:
    joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle, 
                  elbow_angle]. Angles are in radians
  Returns:
    4x4 matrix representing the pose of the shoulder frame in the base frame
  """

  joint_angles = np.asarray(joint_angles, dtype=np.float64).reshape(3)
  shoulder_frame = fk_hip(joint_angles) @ homogenous_transformation_matrix(
    axis=np.array([0.0, 1.0, 0.0]),
    angle=joint_angles[1],
    v_A=np.array([0.0, -HIP_OFFSET, 0.0]),
  )
  return shoulder_frame

def fk_elbow(joint_angles):
  """
  Use forward kinematics equations to calculate the xyz coordinates of the elbow
  joint given the joint angles of the robot

  Args:
    joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle, 
                  elbow_angle]. Angles are in radians
  Returns:
    4x4 matrix representing the pose of the elbow frame in the base frame
  """

  joint_angles = np.asarray(joint_angles, dtype=np.float64).reshape(3)
  elbow_frame = fk_shoulder(joint_angles) @ homogenous_transformation_matrix(
    axis=np.array([0.0, 1.0, 0.0]),
    angle=joint_angles[2],
    v_A=np.array([0.0, 0.0, UPPER_LEG_OFFSET]),
  )
  return elbow_frame

def fk_foot(joint_angles):
  """
  Use forward kinematics equations to calculate the xyz coordinates of the foot given 
  the joint angles of the robot

  Args:
    joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle, 
                  elbow_angle]. Angles are in radians
  Returns:
    4x4 matrix representing the pose of the end effector frame in the base frame
  """

  joint_angles = np.asarray(joint_angles, dtype=np.float64).reshape(3)
  end_effector_frame = fk_elbow(joint_angles) @ homogenous_transformation_matrix(
    axis=np.array([0.0, 1.0, 0.0]),
    angle=0.0,
    v_A=np.array([0.0, 0.0, LOWER_LEG_OFFSET]),
  )
  return end_effector_frame
