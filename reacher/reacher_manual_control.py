import time
from pathlib import Path
import sys

import pybullet as p
import numpy as np
np.set_printoptions(suppress=True, precision=3)

from absl import app
from absl import flags

if __package__ in (None, ""):
	sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
	from reacher import forward_kinematics
	from reacher import inverse_kinematics
	from reacher.joint_conventions import real_to_sim_joint_positions
	from reacher.joint_conventions import sim_to_real_joint_positions
	from reacher import reacher_sim_utils
	from reacher.dynamixel_interface import Reacher
else:
	from . import forward_kinematics
	from . import inverse_kinematics
	from .joint_conventions import real_to_sim_joint_positions
	from .joint_conventions import sim_to_real_joint_positions
	from . import reacher_sim_utils
	from .dynamixel_interface import Reacher


flags.DEFINE_bool("run_on_robot", False, "Whether to run on robot or in simulation.")
flags.DEFINE_bool("ik"          , False, "Whether to control arms through cartesian coordinates(IK) or joint angles")
flags.DEFINE_list("set_joint_positions", [], "List of joint angles to set at initialization.")
flags.DEFINE_bool("real_to_sim", False, "Whether we want to command the sim robot by moving the real robot")
flags.DEFINE_bool("sim_to_real", False, "Whether we want to command the real robot by moving the sim robot (by moving the slider)")
FLAGS = flags.FLAGS

UPDATE_DT = 0.01  # seconds

def main(argv):
	run_on_robot = FLAGS.run_on_robot
	reacher_sim = reacher_sim_utils.load_reacher()

	# Sphere markers for the students' FK solutions
	shoulder_sphere_id = reacher_sim_utils.create_debug_sphere([1, 0, 0, 1])
	elbow_sphere_id    = reacher_sim_utils.create_debug_sphere([0, 1, 0, 1])
	foot_sphere_id     = reacher_sim_utils.create_debug_sphere([0, 0, 1, 1])
	target_sphere_id   = reacher_sim_utils.create_debug_sphere([1, 1, 1, 1], radius=0.01)

	joint_ids = reacher_sim_utils.get_joint_ids(reacher_sim)
	param_ids = reacher_sim_utils.get_param_ids(reacher_sim, FLAGS.ik)
	reacher_sim_utils.zero_damping(reacher_sim)

	p.setPhysicsEngineParameter(numSolverIterations=10)

	# Set up physical robot if we're using it
	sim_to_real = False
	real_to_sim = False
	startup_real_joint_positions = None
	if run_on_robot:
		reacher_real = Reacher()
		time.sleep(0.25)
		startup_real_joint_positions = reacher_real.get_joint_positions()
		print("Treating the robot's startup base angle as the simulator's forward direction.")
		real_to_sim = FLAGS.real_to_sim
		sim_to_real = FLAGS.sim_to_real
		if real_to_sim and sim_to_real:
			raise ValueError("Please choose only one of --real_to_sim or --sim_to_real.")
		if real_to_sim:
			reacher_real.disable_torque()
			print("Real-to-sim mode enabled: motor torque disabled so you can move the robot by hand.")
		else:
			reacher_real.enable_torque()
		if sim_to_real:
			print("Sim-to-real mode enabled: slider commands will be sent to the real robot.")
		if not (real_to_sim or sim_to_real):
			print("Robot connected, but neither --real_to_sim nor --sim_to_real was set, so the robot will only be monitored.")
		# reacher_real.reset()

	# Control Loop Variables
	p.setRealTimeSimulation(1)
	counter = 0
	last_command = time.time()
	joint_positions = np.zeros(3)

	if flags.FLAGS.set_joint_positions:
		# First set the joint angles to 0,0,0
		for idx, joint_id in enumerate(joint_ids):
			p.setJointMotorControl2(
			reacher_sim,
			joint_id,
			p.POSITION_CONTROL,
			joint_positions[idx],
			force=2.
			)
		joint_positions = np.array(flags.FLAGS.set_joint_positions, dtype=np.float32)
		# Set the simulated robot to match the joint angles
		for idx, joint_id in enumerate(joint_ids):
			p.setJointMotorControl2(
			reacher_sim,
			joint_id,
			p.POSITION_CONTROL,
			joint_positions[idx],
			force=2.
			)


	# Main loop
	while (True):

		# Control loop
		if time.time() - last_command > UPDATE_DT:
			last_command = time.time()
			counter += 1

			# Read the slider values
			try:
				slider_values = np.array([p.readUserDebugParameter(id) for id in param_ids])
			except:
				pass
			if FLAGS.ik:
				xyz = slider_values
				p.resetBasePositionAndOrientation(target_sphere_id, posObj=xyz, ornObj=[0, 0, 0, 1])
			else:
				sim_target_joint_positions = slider_values

			# Read the simulated robot's joint angles
			sim_joint_positions = []
			for idx, joint_id in enumerate(joint_ids):
				sim_joint_positions.append(p.getJointState(reacher_sim, joint_id)[0])
			sim_joint_positions = np.array(sim_joint_positions)
			
			
			# If IK is enabled, update joint angles based off of goal XYZ position
			if FLAGS.ik:
				ret = inverse_kinematics.calculate_inverse_kinematics(xyz, sim_joint_positions[:3])
				if ret is not None:
					enable = True
					# Wraps angles between -pi, pi
					sim_target_joint_positions = np.arctan2(np.sin(ret), np.cos(ret))

					# Double check that the angles are a correct solution before sending anything to the real robot
					# If the error between the goal foot position and the position of the foot obtained from the IK solution is too large,
					# don't set the joint angles of the robot to the angles obtained from IK 
					pos = forward_kinematics.fk_foot(sim_target_joint_positions[:3])[:3,3]
					if np.linalg.norm(np.asarray(pos) - xyz) > 0.05:
						sim_target_joint_positions = np.zeros_like(sim_target_joint_positions)
						if flags.FLAGS.set_joint_positions:
							sim_target_joint_positions = np.array(flags.FLAGS.set_joint_positions, dtype=np.float32)
						print("Prevented operation on real robot as inverse kinematics solution was not correct")

			# If real-to-sim, update the simulated robot's joint angles based on the real robot's joint angles
			if real_to_sim:
				sim_target_joint_positions = real_to_sim_joint_positions(
					reacher_real.get_joint_positions(),
					startup_real_joint_positions,
				)

			# Set the simulated robot's joint positions to sim_target_joint_positions
			for idx, joint_id in enumerate(joint_ids):
				p.setJointMotorControl2(
					reacher_sim,
					joint_id,
					p.POSITION_CONTROL,
					sim_target_joint_positions[idx],
					force=2.
				)

			# If sim-to-real, update the real robot's joint angles based on the simulated robot's joint angle
			if sim_to_real:
				target_joint_positions = sim_to_real_joint_positions(
					sim_joint_positions,
					startup_real_joint_positions,
				)
				reacher_real.set_joint_positions(target_joint_positions)
			
			
			# Obtain the real robot's joint angles
			if run_on_robot:
				real_joint_positions = reacher_real.get_joint_positions()


			# Get the calculated positions of each joint and the end effector
			shoulder_pos = forward_kinematics.fk_shoulder(sim_joint_positions[:3])[:3,3]
			elbow_pos    = forward_kinematics.fk_elbow(sim_joint_positions[:3])[:3,3]
			foot_pos     = forward_kinematics.fk_foot(sim_joint_positions[:3])[:3,3]

			# Show the debug spheres for FK
			p.resetBasePositionAndOrientation(shoulder_sphere_id, posObj=shoulder_pos, ornObj=[0, 0, 0, 1])
			p.resetBasePositionAndOrientation(elbow_sphere_id   , posObj=elbow_pos   , ornObj=[0, 0, 0, 1])
			p.resetBasePositionAndOrientation(foot_sphere_id    , posObj=foot_pos    , ornObj=[0, 0, 0, 1])

			# This is a small hack. Ignore this.
			if flags.FLAGS.set_joint_positions and counter == 1:
				time.sleep(2)

			# Show the result in the terminal
			if counter % 20 == 0:
				print("Simulated robot joint positions: ", sim_joint_positions)
				if run_on_robot:
					print("Real robot joint positions: ", reacher_real.get_joint_positions())

app.run(main)
