import time

from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera
from lerobot.common.robot_devices.motors.ufactory import XArmWrapper
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot

# Defines how to communicate with the motors of the leader and follower arms
leader_arms = {
    "main": XArmWrapper(
        port="192.168.1.236",
        motors={
            # name: (index, model)
            "joint1": (1, "ufactory-850"),
            "joint2": (2, "ufactory-850"),
            "joint3": (3, "ufactory-850"),
            "joint4": (4, "ufactory-850"),
            "joint5": (5, "ufactory-850"),
            "joint6": (6, "ufactory-850"),
            "gripper": (7, "ufactory-850"),
        },
    ),
}
follower_arms = {
    "main": XArmWrapper(
        port="192.168.1.218",
        motors={
            # name: (index, model)
            "joint1": (1, "ufactory-850"),
            "joint2": (2, "ufactory-850"),
            "joint3": (3, "ufactory-850"),
            "joint4": (4, "ufactory-850"),
            "joint5": (5, "ufactory-850"),
            "joint6": (6, "ufactory-850"),
            "gripper": (7, "ufactory-850"),
        },
    ),
}
robot = ManipulatorRobot(
    robot_type="u850",
    calibration_dir=".cache/calibration/u850",
    leader_arms=leader_arms,
    follower_arms=follower_arms,
    cameras={
        "top": OpenCVCamera(4, fps=30, width=640, height=480),
        "wrist": OpenCVCamera(10, fps=30, width=640, height=480),
    },
)

# Connect motors buses and cameras if any (Required)
robot.connect()

try:
    import time

    start_time = time.time()
    iteration_count = 0

    while True:
        robot.teleop_step()
        # time.sleep(0.033)  # 30 Hz -> barely smooth
        # time.sleep(0.004)  # 250 Hz -> very smooth

        iteration_count += 1
        elapsed_time = time.time() - start_time

        if elapsed_time >= 1.0:  # Print frequency every second
            frequency = iteration_count / elapsed_time
            print(f"Current teleoperation frequency: {frequency:.2f} Hz")
            start_time = time.time()
            iteration_count = 0

        # # Recording data, only joints
        # leader_pos = robot.leader_arms["main"].get_position()
        # follower_pos = robot.follower_arms["main"].get_position()
        # observation, action = robot.teleop_step(record_data=True)

        # print(follower_pos)
        # print(observation)
        # print(leader_pos)
        # print(action)
        # print("---")

        # Recording data with cameras
        # observation, action = robot.teleop_step(record_data=True)
        # print(observation["observation.images.top"].shape)
        # print(observation["observation.images.wrist"].shape)
        # print(observation["observation.images.top"].min().item())
        # print(observation["observation.images.top"].max().item())
        # print("---")


except KeyboardInterrupt:
    print("Operation interrupted by user.")

# seconds = 30
# frequency = 200
# for _ in tqdm.tqdm(range(seconds*frequency)):
#     robot.teleop_step()

robot.disconnect()
