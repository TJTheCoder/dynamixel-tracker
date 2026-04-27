"""Track a single raised fingertip and move the robot relative to a calibrated start pose.

The first time the script sees exactly one non-thumb finger held straight up, it
records that fingertip location as the calibration point. The robot's current
pose at startup is treated as the matching "straight up" arm pose. After that,
moving the fingertip across the webcam image moves the robot end effector by a
similar relative amount in Cartesian space.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from reacher import forward_kinematics
    from reacher import inverse_kinematics
    from reacher.joint_conventions import real_to_sim_joint_positions
    from reacher.joint_conventions import sim_to_real_joint_positions
else:
    from . import forward_kinematics
    from . import inverse_kinematics
    from .joint_conventions import real_to_sim_joint_positions
    from .joint_conventions import sim_to_real_joint_positions


FLAGS = None

TIP_IDS = {
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}

PIP_IDS = {
    "index": 6,
    "middle": 10,
    "ring": 14,
    "pinky": 18,
}

MCP_IDS = {
    "index": 5,
    "middle": 9,
    "ring": 13,
    "pinky": 17,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate on one upright finger, then move the robot arm with fingertip motion."
    )
    parser.add_argument("--run_on_robot", dest="run_on_robot", action="store_true", default=True)
    parser.add_argument("--no-run_on_robot", dest="run_on_robot", action="store_false")
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--camera_width", type=int, default=1280)
    parser.add_argument("--camera_height", type=int, default=720)
    parser.add_argument("--mirror_camera", dest="mirror_camera", action="store_true", default=True)
    parser.add_argument("--no-mirror_camera", dest="mirror_camera", action="store_false")
    parser.add_argument("--invert_x", dest="invert_x", action="store_true", default=False)
    parser.add_argument("--no-invert_x", dest="invert_x", action="store_false")
    parser.add_argument("--invert_z", dest="invert_z", action="store_true", default=False)
    parser.add_argument("--no-invert_z", dest="invert_z", action="store_false")
    parser.add_argument(
        "--x_gain",
        type=float,
        default=0.18,
        help="Meters of robot X motion per 1.0 of normalized screen X motion.",
    )
    parser.add_argument(
        "--z_gain",
        type=float,
        default=0.18,
        help="Meters of robot Z motion per 1.0 of normalized screen Y motion.",
    )
    parser.add_argument("--target_y_offset", type=float, default=0.0)
    parser.add_argument("--workspace_x_min", type=float, default=-0.14)
    parser.add_argument("--workspace_x_max", type=float, default=0.14)
    parser.add_argument("--workspace_z_min", type=float, default=0.03)
    parser.add_argument("--workspace_z_max", type=float, default=0.23)
    parser.add_argument("--target_smoothing", type=float, default=0.2)
    parser.add_argument("--command_deadband", type=float, default=0.004)
    parser.add_argument("--max_target_step", type=float, default=0.025)
    parser.add_argument("--max_ik_error", type=float, default=0.03)
    parser.add_argument("--command_rate_hz", type=float, default=20.0)
    parser.add_argument("--min_detection_confidence", type=float, default=0.6)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.6)
    parser.add_argument("--calibration_upright_threshold_deg", type=float, default=25.0)
    parser.add_argument("--calibration_straight_threshold_deg", type=float, default=25.0)
    parser.add_argument(
        "--calibration_center_tolerance_norm",
        type=float,
        default=0.05,
        help="Maximum normalized fingertip distance from screen center for calibration.",
    )
    parser.add_argument(
        "--calibration_hold_frames",
        type=int,
        default=8,
        help="Number of consecutive centered, upright frames required to calibrate.",
    )
    return parser.parse_args()


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def vector_clamp_step(previous: np.ndarray, current: np.ndarray, max_step: float) -> np.ndarray:
    delta = current - previous
    distance = np.linalg.norm(delta)
    if distance <= max_step or distance < 1e-9:
        return current
    return previous + (delta / distance) * max_step


def landmark_xy(hand_landmarks, landmark_id: int) -> np.ndarray:
    landmark = hand_landmarks.landmark[landmark_id]
    return np.array([landmark.x, landmark.y], dtype=np.float64)


def signed_angle_2d(reference: np.ndarray, vector: np.ndarray) -> float | None:
    ref_norm = np.linalg.norm(reference)
    vec_norm = np.linalg.norm(vector)
    if ref_norm < 1e-9 or vec_norm < 1e-9:
        return None

    ref = reference / ref_norm
    vec = vector / vec_norm
    cross_z = ref[0] * vec[1] - ref[1] * vec[0]
    dot = np.clip(np.dot(ref, vec), -1.0, 1.0)
    return math.atan2(cross_z, dot)


def get_single_raised_finger(hand_landmarks) -> str | None:
    extended_fingers = []

    for name, tip_id in TIP_IDS.items():
        pip_id = PIP_IDS[name]
        mcp_id = MCP_IDS[name]
        tip = hand_landmarks.landmark[tip_id]
        pip = hand_landmarks.landmark[pip_id]
        mcp = hand_landmarks.landmark[mcp_id]

        tip_above_pip = tip.y < pip.y - 0.03
        pip_above_mcp = pip.y < mcp.y - 0.015
        if tip_above_pip and pip_above_mcp:
            extended_fingers.append(name)

    if len(extended_fingers) != 1:
        return None
    return extended_fingers[0]


def normalized_hand_size(hand_landmarks) -> float:
    xs = [landmark.x for landmark in hand_landmarks.landmark]
    ys = [landmark.y for landmark in hand_landmarks.landmark]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def calibration_metrics(hand_landmarks, finger_name: str) -> tuple[float | None, float | None]:
    mcp = landmark_xy(hand_landmarks, MCP_IDS[finger_name])
    pip = landmark_xy(hand_landmarks, PIP_IDS[finger_name])
    tip = landmark_xy(hand_landmarks, TIP_IDS[finger_name])

    full_finger = tip - mcp
    proximal = pip - mcp
    distal = tip - pip

    upright_angle = signed_angle_2d(np.array([0.0, -1.0], dtype=np.float64), full_finger)
    straight_angle = signed_angle_2d(proximal, distal)
    if upright_angle is None or straight_angle is None:
        return None, None
    return abs(math.degrees(upright_angle)), abs(math.degrees(straight_angle))


def is_calibration_pose(hand_landmarks, finger_name: str) -> bool:
    upright_error_deg, straight_error_deg = calibration_metrics(hand_landmarks, finger_name)
    if upright_error_deg is None or straight_error_deg is None:
        return False

    return (
        upright_error_deg <= FLAGS.calibration_upright_threshold_deg
        and straight_error_deg <= FLAGS.calibration_straight_threshold_deg
    )


def center_distance_norm(hand_landmarks, finger_name: str) -> float:
    fingertip_xy = landmark_xy(hand_landmarks, TIP_IDS[finger_name])
    return float(np.linalg.norm(fingertip_xy - np.array([0.5, 0.5], dtype=np.float64)))


def fingertip_to_target(
    hand_landmarks,
    finger_name: str,
    calibration_tip_xy: np.ndarray,
    neutral_robot_xyz: np.ndarray,
) -> tuple[np.ndarray, float]:
    fingertip_xy = landmark_xy(hand_landmarks, TIP_IDS[finger_name])
    hand_size = normalized_hand_size(hand_landmarks)
    fingertip_delta = fingertip_xy - calibration_tip_xy

    x_direction = -1.0 if FLAGS.invert_x else 1.0
    z_direction = 1.0 if FLAGS.invert_z else -1.0

    target_x = neutral_robot_xyz[0] + x_direction * fingertip_delta[0] * FLAGS.x_gain
    target_z = neutral_robot_xyz[2] + z_direction * fingertip_delta[1] * FLAGS.z_gain

    target_x = clamp(target_x, FLAGS.workspace_x_min, FLAGS.workspace_x_max)
    target_z = clamp(target_z, FLAGS.workspace_z_min, FLAGS.workspace_z_max)

    target_xyz = np.array(
        [target_x, neutral_robot_xyz[1] + FLAGS.target_y_offset, target_z],
        dtype=np.float64,
    )
    return target_xyz, hand_size


def solve_ik(target_xyz: np.ndarray, guess: np.ndarray) -> tuple[np.ndarray | None, float]:
    solution = inverse_kinematics.calculate_inverse_kinematics(target_xyz, guess)
    solution = np.arctan2(np.sin(solution), np.cos(solution))
    reconstructed_xyz = forward_kinematics.fk_foot(solution[:3])[:3, 3]
    error = float(np.linalg.norm(reconstructed_xyz - target_xyz))
    if not np.all(np.isfinite(solution)) or error > FLAGS.max_ik_error:
        return None, error
    return solution, error


def put_status_text(cv2, frame, lines: list[str]) -> None:
    y = 30
    for line in lines:
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (40, 230, 40),
            2,
            cv2.LINE_AA,
        )
        y += 28


def draw_fingerprint_target(cv2, frame, is_ready: bool, hold_progress: float) -> None:
    height, width = frame.shape[:2]
    center = (width // 2, height // 2)
    color = (40, 230, 40) if is_ready else (220, 220, 220)
    radius = max(24, min(width, height) // 18)

    cv2.circle(frame, center, 5, color, thickness=-1)
    cv2.circle(frame, center, radius + 18, color, thickness=1, lineType=cv2.LINE_AA)
    cv2.line(frame, (center[0] - radius - 28, center[1]), (center[0] - radius - 8, center[1]), color, 2, cv2.LINE_AA)
    cv2.line(frame, (center[0] + radius + 8, center[1]), (center[0] + radius + 28, center[1]), color, 2, cv2.LINE_AA)
    cv2.line(frame, (center[0], center[1] - radius - 28), (center[0], center[1] - radius - 8), color, 2, cv2.LINE_AA)
    cv2.line(frame, (center[0], center[1] + radius + 8), (center[0], center[1] + radius + 28), color, 2, cv2.LINE_AA)

    # A simple fingerprint-like marker made of nested arcs.
    for scale in (1.0, 0.72, 0.46):
        axes = (int(radius * scale), int(radius * 1.25 * scale))
        cv2.ellipse(frame, center, axes, 0, 25, 335, color, 2, cv2.LINE_AA)
    cv2.ellipse(
        frame,
        (center[0], center[1] + int(radius * 0.12)),
        (int(radius * 0.28), int(radius * 0.36)),
        0,
        0,
        360,
        color,
        2,
        cv2.LINE_AA,
    )

    if hold_progress > 0.0:
        sweep = int(360 * clamp(hold_progress, 0.0, 1.0))
        cv2.ellipse(
            frame,
            center,
            (radius + 10, radius + 10),
            -90,
            0,
            sweep,
            (80, 190, 255),
            3,
            cv2.LINE_AA,
        )


def draw_fingertip_marker(cv2, frame, hand_landmarks, finger_name: str) -> None:
    height, width = frame.shape[:2]
    fingertip_xy = landmark_xy(hand_landmarks, TIP_IDS[finger_name])
    pixel = (int(fingertip_xy[0] * width), int(fingertip_xy[1] * height))
    cv2.circle(frame, pixel, 8, (80, 190, 255), thickness=2, lineType=cv2.LINE_AA)


def validate_workspace() -> None:
    if FLAGS.workspace_x_min >= FLAGS.workspace_x_max:
        raise ValueError("workspace_x_min must be smaller than workspace_x_max.")
    if FLAGS.workspace_z_min >= FLAGS.workspace_z_max:
        raise ValueError("workspace_z_min must be smaller than workspace_z_max.")
    if FLAGS.command_rate_hz <= 0.0:
        raise ValueError("command_rate_hz must be positive.")
    if not 0.0 < FLAGS.target_smoothing <= 1.0:
        raise ValueError("target_smoothing must be in the interval (0, 1].")
    if FLAGS.calibration_hold_frames <= 0:
        raise ValueError("calibration_hold_frames must be positive.")


def open_camera(cv2):
    backends = []
    if sys.platform.startswith("win") and hasattr(cv2, "CAP_DSHOW"):
        backends.append(cv2.CAP_DSHOW)
    backends.append(None)

    for backend in backends:
        if backend is None:
            capture = cv2.VideoCapture(FLAGS.camera_index)
        else:
            capture = cv2.VideoCapture(FLAGS.camera_index, backend)

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, FLAGS.camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FLAGS.camera_height)
        if capture.isOpened():
            return capture
        capture.release()

    raise RuntimeError(
        f"Could not open webcam index {FLAGS.camera_index}. "
        "Try another index with --camera_index."
    )


def mediapipe_modules(mp):
    if hasattr(mp, "solutions"):
        return mp.solutions.hands, mp.solutions.drawing_utils, mp.solutions.drawing_styles

    raise RuntimeError(
        "This script expects the classic MediaPipe Hands API. "
        "Install the pinned version from requirements_finger_follow.txt."
    )


def send_robot_joint_positions(
    reacher_real,
    sim_joint_positions: np.ndarray,
    startup_real_joint_positions: np.ndarray,
) -> None:
    real_joint_target = sim_to_real_joint_positions(
        sim_joint_positions,
        startup_real_joint_positions,
    )
    reacher_real.set_joint_positions(real_joint_target)


def main():
    validate_workspace()

    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise ImportError(
            "This script requires the webcam packages from requirements_finger_follow.txt. "
            "Install them with `pip install -r requirements_finger_follow.txt`."
        ) from exc

    hands_module, drawing, drawing_styles = mediapipe_modules(mp)

    reacher_real = None
    startup_real_joint_positions = None
    sim_joint_guess = np.zeros(3, dtype=np.float64)

    if FLAGS.run_on_robot:
        if __package__ in (None, ""):
            from reacher.dynamixel_interface import Reacher
        else:
            from .dynamixel_interface import Reacher

        reacher_real = Reacher()
        time.sleep(0.25)
        reacher_real.enable_torque()
        startup_real_joint_positions = reacher_real.get_joint_positions()
        sim_joint_guess = real_to_sim_joint_positions(
            startup_real_joint_positions,
            startup_real_joint_positions,
        )
        print("Treating the robot's startup base angle as the forward-facing reference.")

    neutral_joint_positions = sim_joint_guess.copy()
    neutral_robot_xyz = forward_kinematics.fk_foot(neutral_joint_positions[:3])[:3, 3]
    smoothed_target = neutral_robot_xyz.copy()
    last_commanded_target = neutral_robot_xyz.copy()

    calibration_tip_xy = None
    calibration_finger_name = None
    calibration_hold_counter = 0

    capture = open_camera(cv2)

    print("Press 'q' in the webcam window to quit.")
    print("Press 'r' to recalibrate.")
    print("Hold exactly one non-thumb finger straight up and center the fingertip on the target.")
    print("The robot's startup pose is treated as the matching straight-up pose.")

    command_period = 1.0 / FLAGS.command_rate_hz
    last_command_time = 0.0

    with hands_module.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=FLAGS.min_detection_confidence,
        min_tracking_confidence=FLAGS.min_tracking_confidence,
    ) as hands:
        try:
            while True:
                frame_ok, frame = capture.read()
                if not frame_ok:
                    raise RuntimeError("Webcam frame capture failed.")

                if FLAGS.mirror_camera:
                    frame = cv2.flip(frame, 1)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                result = hands.process(frame_rgb)

                status_lines = ["Hold one finger straight up to calibrate."]
                target_ready = False
                hold_progress = calibration_hold_counter / FLAGS.calibration_hold_frames

                if result.multi_hand_landmarks:
                    hand_landmarks = result.multi_hand_landmarks[0]
                    drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        hands_module.HAND_CONNECTIONS,
                        drawing_styles.get_default_hand_landmarks_style(),
                        drawing_styles.get_default_hand_connections_style(),
                    )

                    finger_name = get_single_raised_finger(hand_landmarks)
                    if finger_name is None:
                        calibration_hold_counter = 0
                        status_lines = ["Raise exactly one non-thumb finger."]
                    elif calibration_tip_xy is None:
                        draw_fingertip_marker(cv2, frame, hand_landmarks, finger_name)
                        upright_error_deg, straight_error_deg = calibration_metrics(
                            hand_landmarks, finger_name
                        )
                        center_error = center_distance_norm(hand_landmarks, finger_name)
                        pose_ok = is_calibration_pose(hand_landmarks, finger_name)
                        center_ok = center_error <= FLAGS.calibration_center_tolerance_norm
                        target_ready = pose_ok and center_ok
                        if target_ready:
                            calibration_hold_counter += 1
                        else:
                            calibration_hold_counter = 0

                        hold_progress = calibration_hold_counter / FLAGS.calibration_hold_frames
                        if calibration_hold_counter >= FLAGS.calibration_hold_frames:
                            calibration_tip_xy = landmark_xy(hand_landmarks, TIP_IDS[finger_name])
                            calibration_finger_name = finger_name
                            calibration_hold_counter = 0
                            smoothed_target = neutral_robot_xyz.copy()
                            last_commanded_target = neutral_robot_xyz.copy()
                            status_lines = [
                                f"Calibrated on {finger_name} finger.",
                                "Move your fingertip across the screen.",
                            ]
                        else:
                            if target_ready:
                                frames_left = FLAGS.calibration_hold_frames - calibration_hold_counter
                                status_lines = [
                                    "Hold steady on the center target.",
                                    f"calibrating in {frames_left} frame(s)",
                                ]
                            else:
                                status_lines = [
                                    "Hold one finger straight up and center the fingertip.",
                                    (
                                        f"center error = {center_error:.3f}, "
                                        f"upright error = {0.0 if upright_error_deg is None else upright_error_deg:.1f} deg, "
                                        f"bend error = {0.0 if straight_error_deg is None else straight_error_deg:.1f} deg"
                                    ),
                                ]
                    elif finger_name != calibration_finger_name:
                        calibration_hold_counter = 0
                        status_lines = [
                            f"Keep using the {calibration_finger_name} finger.",
                            "Press 'r' to recalibrate on a different finger.",
                        ]
                    else:
                        draw_fingertip_marker(cv2, frame, hand_landmarks, finger_name)
                        raw_target, hand_size = fingertip_to_target(
                            hand_landmarks,
                            finger_name,
                            calibration_tip_xy,
                            neutral_robot_xyz,
                        )
                        filtered_target = (
                            (1.0 - FLAGS.target_smoothing) * smoothed_target
                            + FLAGS.target_smoothing * raw_target
                        )
                        smoothed_target = vector_clamp_step(
                            smoothed_target, filtered_target, FLAGS.max_target_step
                        )

                        now = time.time()
                        target_delta = np.linalg.norm(smoothed_target - last_commanded_target)
                        if target_delta >= FLAGS.command_deadband and now - last_command_time >= command_period:
                            ik_solution, ik_error = solve_ik(smoothed_target, sim_joint_guess)
                            if ik_solution is not None:
                                sim_joint_guess = ik_solution
                                last_commanded_target = smoothed_target.copy()
                                last_command_time = now

                                if reacher_real is not None:
                                    send_robot_joint_positions(
                                        reacher_real,
                                        sim_joint_guess,
                                        startup_real_joint_positions,
                                    )

                                status_lines = [
                                    f"Tracking {finger_name} finger",
                                    (
                                        f"xyz = {smoothed_target[0]:+.3f}, "
                                        f"{smoothed_target[1]:+.3f}, {smoothed_target[2]:+.3f} m"
                                    ),
                                    f"hand size = {hand_size:.3f}, IK error = {ik_error:.3f} m",
                                ]
                            else:
                                status_lines = [
                                    "IK target rejected for safety.",
                                    (
                                        f"xyz = {smoothed_target[0]:+.3f}, "
                                        f"{smoothed_target[1]:+.3f}, {smoothed_target[2]:+.3f} m"
                                    ),
                                    f"Try a smaller fingertip motion. IK error = {ik_error:.3f} m",
                                ]
                        else:
                            status_lines = [
                                f"Tracking {finger_name} finger",
                                (
                                    f"xyz = {smoothed_target[0]:+.3f}, "
                                    f"{smoothed_target[1]:+.3f}, {smoothed_target[2]:+.3f} m"
                                ),
                                f"hand size = {hand_size:.3f}",
                            ]
                else:
                    calibration_hold_counter = 0
                    if calibration_tip_xy is not None and reacher_real is not None:
                        status_lines = ["Hand not detected. Holding current robot pose."]

                if calibration_tip_xy is None:
                    draw_fingerprint_target(
                        cv2,
                        frame,
                        is_ready=target_ready,
                        hold_progress=hold_progress,
                    )

                put_status_text(cv2, frame, status_lines)
                cv2.imshow("Finger Follow Controller", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("r"):
                    calibration_tip_xy = None
                    calibration_finger_name = None
                    calibration_hold_counter = 0
                    smoothed_target = neutral_robot_xyz.copy()
                    last_commanded_target = neutral_robot_xyz.copy()
                    sim_joint_guess = neutral_joint_positions.copy()
                    if reacher_real is not None:
                        send_robot_joint_positions(
                            reacher_real,
                            sim_joint_guess,
                            startup_real_joint_positions,
                        )
        finally:
            capture.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    FLAGS = parse_args()
    main()
