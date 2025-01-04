# Demonstrate a simple behavior of the robot programmed using UFACTORY's SDK and tools

import sys
import math
import time
import queue
import datetime
import random
import traceback
import threading
from xarm import version
from xarm.wrapper import XArmAPI


class RobotMain(object):
    """Robot Main Class"""
    def __init__(self, robot, **kwargs):
        self.alive = True
        self._arm = robot
        self._ignore_exit_state = False
        self._tcp_speed = 100
        self._tcp_acc = 2000
        self._angle_speed = 20
        self._angle_acc = 500
        self._vars = {}
        self._funcs = {}
        self._robot_init()

    # Robot init
    def _robot_init(self):
        self._arm.clean_warn()
        self._arm.clean_error()
        self._arm.motion_enable(True)
        self._arm.set_mode(0)
        self._arm.set_state(0)
        time.sleep(1)
        self._arm.register_error_warn_changed_callback(self._error_warn_changed_callback)
        self._arm.register_state_changed_callback(self._state_changed_callback)

    def set_manual_mode(self):
        """Set the robot arm to manual mode (mode 2)"""
        self._arm.set_mode(2)  # Mode 2 is manual mode
        self._arm.set_state(0)  # Set state to 0 (ready)

    def set_motion_mode(self):
        """Set the robot arm to motion mode (mode 0)"""
        self._arm.set_mode(0)  # Mode 0 is motion mode
        self._arm.set_state(0)  # Set state to 0 (ready)

    # Register error/warn changed callback
    def _error_warn_changed_callback(self, data):
        if data and data['error_code'] != 0:
            self.alive = False
            self.pprint('err={}, quit'.format(data['error_code']))
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)

    # Register state changed callback
    def _state_changed_callback(self, data):
        if not self._ignore_exit_state and data and data['state'] == 4:
            self.alive = False
            self.pprint('state=4, quit')
            self._arm.release_state_changed_callback(self._state_changed_callback)

    def _check_code(self, code, label):
        if not self.is_alive or code != 0:
            self.alive = False
            ret1 = self._arm.get_state()
            ret2 = self._arm.get_err_warn_code()
            self.pprint('{}, code={}, connected={}, state={}, error={}, ret1={}. ret2={}'.format(label, code, self._arm.connected, self._arm.state, self._arm.error_code, ret1, ret2))
        return self.is_alive

    @staticmethod
    def pprint(*args, **kwargs):
        try:
            stack_tuple = traceback.extract_stack(limit=2)[0]
            print('[{}][{}] {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), stack_tuple[1], ' '.join(map(str, args))))
        except:
            print(*args, **kwargs)

    @property
    def arm(self):
        return self._arm

    @property
    def VARS(self):
        return self._vars

    @property
    def FUNCS(self):
        return self._funcs

    @property
    def is_alive(self):
        if self.alive and self._arm.connected and self._arm.error_code == 0:
            if self._ignore_exit_state:
                return True
            if self._arm.state == 5:
                cnt = 0
                while self._arm.state == 5 and cnt < 5:
                    cnt += 1
                    time.sleep(0.1)
            return self._arm.state < 4
        else:
            return False

    # Robot Main Run
    def run(self, side='left', start_event=None, completion_event=None, trajectory1_event=None, trajectory2_event=None, trajectory1_sync=None, trajectory2_sync=None):
        try:
            if start_event:
                start_event.wait()  # Wait for start signal
                
            code = self._arm.set_gripper_position(850, wait=True, speed=5000, auto_enable=True)
            if not self._check_code(code, 'set_gripper_position'):
                return
            time.sleep(0.2)
            code = self._arm.set_gripper_position(200, wait=True, speed=5000, auto_enable=True)
            if not self._check_code(code, 'set_gripper_position'):
                return
            code = self._arm.playback_trajectory(times=1, filename=f'_1_{side}', wait=True, double_speed=1)
            if not self._check_code(code, 'playback_trajectory'):
                return

            if trajectory1_event:
                trajectory1_event.set()  # Signal completion of first trajectory
                
            # Wait for both arms to complete trajectory1 before continuing
            if trajectory1_sync:
                trajectory1_sync.wait()


            code = self._arm.playback_trajectory(times=1, filename=f'_2_{side}', wait=True, double_speed=1)
            if not self._check_code(code, 'playback_trajectory'):
                return


            # code = self._arm.set_gripper_position(200, wait=True, speed=5000, auto_enable=True)
            # if not self._check_code(code, 'set_gripper_position'):
            #     return

            if trajectory2_event:
                trajectory2_event.set()  # Signal completion of second trajectory

            # Wait for both arms to complete trajectory2 before continuing
            if trajectory2_sync:
                trajectory2_sync.wait()

            code = self._arm.set_gripper_position(0, wait=True, speed=5000, auto_enable=True)
            if not self._check_code(code, 'set_gripper_position'):
                return
            time.sleep(0.2)

            # code = self._arm.playback_trajectory(times=1, filename=f'exit{"_right" if side=="right" else ""}', wait=True, double_speed=1)
            # if not self._check_code(code, 'playback_trajectory'):
            #     return

            if completion_event:
                completion_event.set()  # Signal completion

        except Exception as e:
            self.pprint('MainException: {}'.format(e))
        finally:
            self.alive = False
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
            self._arm.release_state_changed_callback(self._state_changed_callback)


if __name__ == '__main__':
    RobotMain.pprint('xArm-Python-SDK Version:{}'.format(version.__version__))

    for _ in range(1):  # Execute 5 times
        # Create synchronization events
        start_event = threading.Event()
        left_completion = threading.Event()
        right_completion = threading.Event()
        left_trajectory1 = threading.Event()
        right_trajectory1 = threading.Event() 
        left_trajectory2 = threading.Event()
        right_trajectory2 = threading.Event()

        # Create barrier events for trajectory synchronization
        trajectory1_sync = threading.Event()
        trajectory2_sync = threading.Event()

        # Initialize both arms
        arm_left = XArmAPI('192.168.1.236', baud_checkset=False)
        arm_right = XArmAPI('192.168.1.218', baud_checkset=False)
        time.sleep(0.5)

        robot_left = RobotMain(arm_left)
        robot_right = RobotMain(arm_right)

        # Create threads for both arms
        left_thread = threading.Thread(target=robot_left.run, args=('left', start_event, left_completion, left_trajectory1, left_trajectory2, trajectory1_sync, trajectory2_sync))
        right_thread = threading.Thread(target=robot_right.run, args=('right', start_event, right_completion, right_trajectory1, right_trajectory2, trajectory1_sync, trajectory2_sync))

        # Start both threads
        left_thread.start()
        right_thread.start()

        # Signal both arms to start
        start_event.set()

        # Wait for both arms to complete first trajectory
        left_trajectory1.wait()
        right_trajectory1.wait()
        # Signal both arms to continue after trajectory1
        trajectory1_sync.set()

        # Wait for both arms to complete second trajectory
        left_trajectory2.wait()
        right_trajectory2.wait()
        # Signal both arms to continue after trajectory2
        trajectory2_sync.set()

        # Wait for both arms to complete
        left_completion.wait()
        right_completion.wait()

        # Wait for threads to finish
        left_thread.join()
        right_thread.join()

        # Set both arms to manual mode before disconnecting
        robot_left.set_manual_mode()
        robot_right.set_manual_mode()

        # Clean up resources before next iteration
        arm_left.disconnect()
        arm_right.disconnect()
        time.sleep(1)  # Brief pause between iterations
