from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
import time

leader_port = "/dev/tty.usbmodem58CD1770801"
follower_port = "/dev/tty.usbmodem58760433121"

leader_arm = FeetechMotorsBus(
    port=leader_port,
    motors={
        # name: (index, model)
        "shoulder_pan": (1, "sts3215"),
        "shoulder_lift": (2, "sts3215"),
        # "elbow_flex": (3, "sts3215"),
        # "wrist_flex": (4, "sts3215"),
        # "wrist_roll": (5, "sts3215"),
        # "gripper": (6, "sts3215"),
    },
)

follower_arm = FeetechMotorsBus(
    port=follower_port,
    motors={
        # name: (index, model)
        "shoulder_pan": (1, "sts3215"),
        "shoulder_lift": (2, "sts3215"),
        # "elbow_flex": (3, "sts3215"),
        # "wrist_flex": (4, "sts3215"),
        # "wrist_roll": (5, "sts3215"),
        # "gripper": (6, "sts3215"),
    },
)

leader_arm.connect()
follower_arm.connect()

leader_pos = leader_arm.read("Present_Position")
follower_pos = follower_arm.read("Present_Position")
print("Leader position:", leader_pos)
print("Follower position:", follower_pos)


from lerobot.common.robot_devices.motors.feetech import TorqueMode
follower_arm.write("Torque_Enable", TorqueMode.ENABLED.value)

time.sleep(1)

# Get the current position
position = follower_arm.read("Present_Position")
magnitude = 200

for motor_idx in range(2):
    position = follower_arm.read("Present_Position")
    print(f"Moving motor {motor_idx} to", position[motor_idx] + magnitude)
    position[motor_idx] += magnitude
    follower_arm.write("Goal_Position", position)
    time.sleep(1)
    print(f"Moving motor {motor_idx} to", position[motor_idx] - magnitude)
    position[motor_idx] -= magnitude
    follower_arm.write("Goal_Position", position)
    time.sleep(1)


leader_arm.disconnect()
follower_arm.disconnect()
