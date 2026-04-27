import math
import numpy as np
import copy

from . import forward_kinematics

HIP_OFFSET = 0.0335
UPPER_LEG_OFFSET = 0.10 # length of link 1
LOWER_LEG_OFFSET = 0.13 # length of link 2
TOLERANCE = 0.01 # tolerance for inverse kinematics
PERTURBATION = 0.0001 # perturbation for finite difference method
MAX_ITERATIONS = 100

def ik_cost(end_effector_pos, guess):
    """Calculates the inverse kinematics cost.

    This function computes the inverse kinematics cost, which represents the Euclidean
    distance between the desired end-effector position and the end-effector position
    resulting from the provided 'guess' joint angles.

    Args:
        end_effector_pos (numpy.ndarray), (3,): The desired XYZ coordinates of the end-effector.
            A numpy array with 3 elements.
        guess (numpy.ndarray), (3,): A guess at the joint angles to achieve the desired end-effector
            position. A numpy array with 3 elements.

    Returns:
        float: The Euclidean distance between end_effector_pos and the calculated end-effector
        position based on the guess.
    """
    end_effector_pos = np.asarray(end_effector_pos, dtype=np.float64).reshape(3)
    guess = np.asarray(guess, dtype=np.float64).reshape(3)
    current_pos = forward_kinematics.fk_foot(guess)[:3, 3]
    cost = np.linalg.norm(end_effector_pos - current_pos)
    
    return cost

def calculate_jacobian_FD(joint_angles, delta):
    """
    Calculate the Jacobian matrix using finite differences.

    This function computes the Jacobian matrix for a given set of joint angles using finite differences.

    Args:
        joint_angles (numpy.ndarray), (3,): The current joint angles. A numpy array with 3 elements.
        delta (float): The perturbation value used to approximate the partial derivatives.

    Returns:
        numpy.ndarray: The Jacobian matrix. A 3x3 numpy array representing the linear mapping
        between joint velocity and end-effector linear velocity.
    """

    joint_angles = np.asarray(joint_angles, dtype=np.float64).reshape(3)
    J = np.zeros((3, 3), dtype=np.float64)
    base_pos = forward_kinematics.fk_foot(joint_angles)[:3, 3]

    for joint_idx in range(3):
        perturbed_joint_angles = joint_angles.copy()
        perturbed_joint_angles[joint_idx] += delta
        perturbed_pos = forward_kinematics.fk_foot(perturbed_joint_angles)[:3, 3]
        J[:, joint_idx] = (perturbed_pos - base_pos) / delta

    return J

def calculate_inverse_kinematics(end_effector_pos, guess):
    """
    Calculate the inverse kinematics solution using the Newton-Raphson method.

    This function iteratively refines a guess for joint angles to achieve a desired end-effector position.
    It uses the Newton-Raphson method along with a finite difference Jacobian to find the solution.

    Args:
        end_effector_pos (numpy.ndarray): The desired XYZ coordinates of the end-effector.
            A numpy array with 3 elements.
        guess (numpy.ndarray): The initial guess for joint angles. A numpy array with 3 elements.

    Returns:
        numpy.ndarray: The refined joint angles that achieve the desired end-effector position.
    """

    end_effector_pos = np.asarray(end_effector_pos, dtype=np.float64).reshape(3)
    guess = np.asarray(guess, dtype=np.float64).reshape(3).copy()

    # Initialize previous cost to infinity
    previous_cost = np.inf
    cost = ik_cost(end_effector_pos, guess)

    for iters in range(MAX_ITERATIONS):
        if cost < TOLERANCE:
            break

        # Calculate the Jacobian matrix using finite differences
        J = calculate_jacobian_FD(guess, PERTURBATION)

        # Calculate the residual
        current_pos = forward_kinematics.fk_foot(guess)[:3, 3]
        residual = end_effector_pos - current_pos

        # Compute the step to update the joint angles using the Moore-Penrose pseudoinverse using numpy.linalg.pinv
        # delta_theta = np.linalg.pinv(J) @ residual
        lambda_sq = 0.001
        J_t = J.T
        delta_theta = np.linalg.solve(J_t @ J + lambda_sq * np.eye(3), J_t @ residual)
        delta_theta = np.clip(delta_theta, -0.5, 0.5)

        # Take a full Newton step to update the guess for joint angles
        improved = False
        LIMITS = [(-1.5, 1.5), (-2.5, 2.5), (-2.5, -2.5)]
        for step_scale in (1.0, 0.5, 0.25, 0.1, 0.05, 0.01):
            candidate_guess = guess + step_scale * delta_theta
            candidate_guess = np.arctan2(np.sin(candidate_guess), np.cos(candidate_guess))
            # for i in range(3):
            #     candidate_guess[i] = np.clip(candidate_guess[i], LIMITS[i][0], LIMITS[i][1])

            candidate_cost = ik_cost(end_effector_pos, candidate_guess)
            if candidate_cost < cost:
                guess = candidate_guess
                cost = candidate_cost
                improved = True
                break

        if not improved:
            break

        if cost < TOLERANCE:
            break
        if abs(previous_cost - cost) < TOLERANCE:
            break
        previous_cost = cost

    return guess
