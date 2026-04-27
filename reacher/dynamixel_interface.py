import pypot
import time
import math
import numpy as np

from pypot.dynamixel.io import Dxl320IO

class Reacher:
    def __init__(self):
        ports = pypot.dynamixel.get_available_ports() 
        print("Found the following ports for the Dynamixel motors: ", ports)
        if len(ports) == 0:
            raise Exception("Unable to scan the dynamicel motor. Please make sure the U2D2 is correctly connected to your machine.")
        port = ports[0]
        self.dxl_io = Dxl320IO(port) 

        self.motor_IDs = self.dxl_io.scan([1, 2, 3])
        self.num_motors = len(self.motor_IDs)
        print(f"Found {self.num_motors} motors with IDs {self.motor_IDs} with current motor positions {self.get_joint_positions()}")
        
    def get_joint_positions(self):
        joint_positions = np.array(self.dxl_io.get_present_position(self.motor_IDs))
        return np.deg2rad(joint_positions)

    def enable_torque(self):
        self.dxl_io.enable_torque(self.motor_IDs)

    def disable_torque(self):
        self.dxl_io.disable_torque(self.motor_IDs)

    def is_torque_enabled(self):
        return self.dxl_io.is_torque_enabled(self.motor_IDs)

    def get_joint_position_by_id(self, id):
        pos = self.dxl_io.get_present_position(id)
        return np.deg2rad(pos)[0]
    
    def set_joint_positions(self, joint_positions):
        joint_positions = np.rad2deg(joint_positions)
        goal_dict = {}
        for i, motor_id in enumerate(self.motor_IDs):
            goal_dict[motor_id] = joint_positions[motor_id - 1]
        self.dxl_io.set_goal_position(goal_dict)
        time.sleep(0.001)

    def set_joint_position_by_id(self, joint_position, id):
        goal_dict = dict()
        goal_dict[id] = joint_position
        self.dxl_io.set_goal_position(goal_dict)
        time.sleep(0.001)

    def reset(self):
        self.set_joint_positions([0.0, 0.0, 0.0])


    
