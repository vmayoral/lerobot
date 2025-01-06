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
from wasabi import color


class RobotMain(object):
    """Robot Main Class"""
    # Gripper position constants
    GRIPPER_OPEN_POS = 800
    GRIPPER_CLOSE_POS = 50

    def __init__(self, robot, name, trajectory_config=None, **kwargs):
        self.name = name
        self.alive = True
        self._arm = robot
        self._ignore_exit_state = False
        self._tcp_speed = 100
        self._tcp_acc = 2000
        self._angle_speed = 20
        self._angle_acc = 500
        self._vars = {}
        self._funcs = {}
        
        # Initialize trajectory configuration
        default_config = {
            'trajectories': [],
            'gripper_positions': []
        }
        self._trajectory_config = trajectory_config if trajectory_config else default_config
        
        # Extract trajectory numbers and gripper positions
        self._trajectory_numbers = self._trajectory_config.get('trajectories', [])[:]  # Make a copy of the list
        self._gripper_positions = self._trajectory_config.get('gripper_positions', [])[:]  # Make a copy of the list
        
        self._current_recording = None  # Track name of trajectory being recorded
        self._current_loaded_trajectory = None  # Track currently loaded trajectory
        
        # Validate configuration
        if len(self._trajectory_numbers) != len(self._gripper_positions):
            raise ValueError(color("Trajectory numbers and gripper positions must have same length", fg="red", bold=True))
        
        # Create dictionary mapping trajectory numbers to their target gripper positions
        self._trajectory_gripper_positions = dict(zip(self._trajectory_numbers, self._gripper_positions))
        
        self._robot_init()

    def set_led(self, state):
        """Set the LED state (on/off)"""
        self._arm.set_tgpio_digital(ionum=2, value=1 if state else 0)

    def blink_led(self, times):
        """Blink LED n times with 200ms on/off periods"""
        for _ in range(times):
            self.set_led(True)
            time.sleep(0.2)
            self.set_led(False)
            if _ < times - 1:  # Don't sleep after last blink
                time.sleep(0.2)

    # Robot init
    def _robot_init(self):
        self._arm.clean_warn()
        self._arm.clean_error()
        self._arm.motion_enable(True)
        self._arm.set_gripper_enable(True) 

        # motion mode
        self.set_motion_mode()

        time.sleep(1)
        self._arm.register_error_warn_changed_callback(self._error_warn_changed_callback)
        self._arm.register_state_changed_callback(self._state_changed_callback)

        self.set_led(False)

    def delete_trajectory(self, name):
        """
        Delete trajectory

        :param name: trajectory name
        :return: code
            code: See the API Code Documentation for details.
        """
        code = self._arm.delete_trajectory(f"{name}.traj")
        
        # If this was the currently loaded trajectory, clear it
        if name == self._current_loaded_trajectory:
            self._current_loaded_trajectory = None
            
        # Remove trajectory number from list if it exists
        try:
            traj_num = int(name.strip('_').split('_')[0])
            if traj_num in self._trajectory_numbers:
                idx = self._trajectory_numbers.index(traj_num)
                self._trajectory_numbers.pop(idx)
                self._gripper_positions.pop(idx)
                self._trajectory_gripper_positions.pop(traj_num)
        except (ValueError, IndexError):
            pass
            
        return code

    def set_manual_mode(self):
        """Set the robot arm to manual mode (mode 2) with error handling and retries"""
        max_retries = 5
        retry_count = 0
        success = False

        while not success and retry_count < max_retries:
            # Clean any existing errors
            self._arm.clean_error()
            self._arm.clean_warn()
            
            # Enable motion
            self._arm.motion_enable(True)
            time.sleep(0.1)
            
            # Try to set manual mode
            mode_code = self._arm.set_mode(2)
            time.sleep(0.1)
            
            # Clean any errors that occurred during mode change
            self._arm.clean_error()
            self._arm.clean_warn()
            
            # Try to set ready state
            state_code = self._arm.set_state(0)
            time.sleep(0.1)
            
            # Verify state
            code, state = self._arm.get_state()
            if code == 0 and state == 2:
                success = True
                break
                    
            retry_count += 1
            time.sleep(0.2)  # Wait before retry
        
        if not success:
            raise RuntimeError(color("Failed to switch to manual mode after max retries", fg="red", bold=True))

    def set_motion_mode(self):
        """Set the robot arm to motion mode (mode 0) with error handling and retries"""

        mode_code = self._arm.set_mode(0)
        state_code = self._arm.set_state(0)

        # max_retries = 5
        # retry_count = 0
        # success = False

        # while not success and retry_count < max_retries:
        #     # Clean any existing errors
        #     self._arm.clean_error()
        #     self._arm.clean_warn()
            
        #     # Enable motion
        #     self._arm.motion_enable(True)
        #     time.sleep(0.1)
            
        #     # Try to set motion mode
        #     mode_code = self._arm.set_mode(0)
        #     time.sleep(0.1)
            
        #     # Clean any errors that occurred during mode change
        #     self._arm.clean_error()
        #     self._arm.clean_warn()
            
        #     # Try to set ready state
        #     state_code = self._arm.set_state(0)
        #     time.sleep(0.1)
            
        #     # Verify state
        #     code, state = self._arm.get_state()
        #     if code == 0 and state == 0:
        #         success = True
        #         break
                    
        #     retry_count += 1
        #     time.sleep(0.2)  # Wait before retry
        
        # if not success:
        #     raise RuntimeError("Failed to switch to motion mode after max retries")

    def start_recording_trajectory(self, trajectory_name):
        """Start recording a new trajectory
        
        Args:
            trajectory_name (str): Name to save the trajectory under
        """
        if self._current_recording is not None:
            raise RuntimeError(color("Already recording a trajectory", fg="red", bold=True))
            
        self._current_recording = trajectory_name
        self._arm.start_record_trajectory()

    def stop_recording_trajectory(self, trajectory_name):
        """Stop recording the current trajectory and save it
        
        Args:
            trajectory_name (str): Name to save the trajectory under - must match start_recording_trajectory
        """
        if self._current_recording is None:
            raise RuntimeError(color("No trajectory being recorded", fg="red", bold=True))
            
        if trajectory_name != self._current_recording:
            raise ValueError(color(f"Trajectory name mismatch: started with {self._current_recording} but stopping with {trajectory_name}", fg="red", bold=True))
            
        self._arm.stop_record_trajectory()
        self._arm.save_record_trajectory(f"{trajectory_name}.traj")
        
        # Extract trajectory number from name and add to list if successful
        try:
            traj_num = int(trajectory_name.strip('_').split('_')[0])  # Extract number from format like "_1_left"
            if traj_num not in self._trajectory_numbers:
                self._trajectory_numbers.append(traj_num)
                # Add default gripper position for new trajectory
                self._gripper_positions.append(self.GRIPPER_CLOSE_POS)
                self._trajectory_gripper_positions[traj_num] = self.GRIPPER_CLOSE_POS
                # Sort both lists based on trajectory numbers
                sorted_indices = sorted(range(len(self._trajectory_numbers)), key=lambda k: self._trajectory_numbers[k])
                self._trajectory_numbers = [self._trajectory_numbers[i] for i in sorted_indices]
                self._gripper_positions = [self._gripper_positions[i] for i in sorted_indices]
        except (ValueError, IndexError):
            self.pprint(color(f"Warning: Could not extract trajectory number from {trajectory_name}", fg="yellow"))
            
        self._current_recording = None
        self.set_motion_mode()

    def load_trajectory(self, trajectory_name):
        """Load a trajectory from file. Only one trajectory can be loaded at a time.
        
        Args:
            trajectory_name (str): Name of trajectory file to load (without .traj extension)
        """
        # If this trajectory is already loaded, no need to reload
        if trajectory_name == self._current_loaded_trajectory:
            return
            
        # Load the new trajectory file
        code = self._arm.load_trajectory(f"{trajectory_name}.traj")
        if not self._check_code(code, 'load_trajectory'):
            raise RuntimeError(color(f"Failed to load trajectory {trajectory_name}", fg="red", bold=True))
            
        self._current_loaded_trajectory = trajectory_name

    def play_trajectory(self, trajectory_name, start_event=None, completion_event=None):
        """Play a trajectory synchronously. Will load the trajectory if not already loaded.
        
        Args:
            trajectory_name (str): Name of trajectory to play
            start_event (threading.Event, optional): Event to wait for before starting
            completion_event (threading.Event, optional): Event to set when complete
        """
        try:
            # Make sure trajectory is loaded
            if trajectory_name != self._current_loaded_trajectory:
                self.load_trajectory(trajectory_name)
                
            # Wait for start signal if provided
            if start_event:
                start_event.wait()
                
            # Play the trajectory
            code = self._arm.playback_trajectory(times=1, filename=trajectory_name, wait=True)
            if not self._check_code(code, 'playback_trajectory'):
                return False
                
            # Signal completion if event provided
            if completion_event:
                completion_event.set()
                
            return True
                
        except Exception as e:
            self.pprint(color(f'PlayTrajectoryException: {e}', fg="red"))
            return False

    def add_new_trajectory(self):
        """Add a new trajectory by starting to record it
        
        NOTE: stop and save should happen outside
        """
        next_traj_num = max(self._trajectory_numbers) + 1 if self._trajectory_numbers else 1
        trajectory_name = f'_{next_traj_num}{self.name}'
        print(color(f"Started recording new trajectory: {trajectory_name}", fg="green"))
        
        self.set_manual_mode()
        self.start_recording_trajectory(trajectory_name)

    def monitor_digital_input(self, stop_event):
        single_click_time = 0.2
        double_click_time = 0.5
        long_click_time = 1.0

        last_press_time = 0
        last_click_time = 0
        long_click_detected = False
        click_count = 0
        long_click_state = False  # starts in motion mode

        while not stop_event.is_set():
            code, value = self._arm.get_tgpio_digital(ionum=2)
            if code == 0:
                current_time = time.time()

                if value == 1:  # Button pressed
                    if last_press_time == 0:
                        last_press_time = current_time
                    elif not long_click_detected and current_time - last_press_time >= long_click_time:
                        long_click_detected = True
                        long_click_state = not long_click_state
                        if long_click_state:
                            self.set_led(True)
                            try:
                                self.add_new_trajectory()
                                print(color("Long click detected -> Started recording a new trajectory", fg="blue"))
                            except RuntimeError as e:
                                print(color(f"Warning: {str(e)}", fg="yellow"))
                        else:
                            self.set_led(False)
                            try:
                                self.stop_recording_trajectory(self._current_recording)
                                print(color("Long click detected -> Stopped recording the current trajectory", fg="blue"))
                            except RuntimeError as e:
                                print(color(f"Warning: {str(e)}", fg="yellow"))
                else:  # Button released
                    if last_press_time != 0:
                        press_duration = current_time - last_press_time

                        if not long_click_detected:
                            if press_duration < single_click_time:
                                click_count += 1
                                if click_count == 1:
                                    last_click_time = current_time
                                elif click_count == 2:
                                    if current_time - last_click_time < double_click_time:
                                        print(color("Double click detected -> Open gripper", fg="blue"))
                                        self._arm.set_gripper_position(pos=self.GRIPPER_OPEN_POS, wait=False)  # Open gripper
                                        click_count = 0
                                    else:
                                        print(color("Single click detected -> Close gripper", fg="blue"))
                                        self._arm.set_gripper_position(pos=self.GRIPPER_CLOSE_POS, wait=False)  # Close gripper
                                        click_count = 1
                                        last_click_time = current_time
                            else:
                                print(color("Single click detected -> Close gripper", fg="blue"))
                                self._arm.set_gripper_position(pos=self.GRIPPER_CLOSE_POS, wait=False)  # Close gripper
                                click_count = 0

                        last_press_time = 0
                        long_click_detected = False

                # Reset click count if too much time has passed since last click
                if click_count == 1 and current_time - last_click_time >= double_click_time:
                    print(color("Single click detected -> Close gripper", fg="blue"))
                    self._arm.set_gripper_position(pos=self.GRIPPER_CLOSE_POS, wait=False)  # Close gripper
                    click_count = 0

            time.sleep(0.01)  # Check every 10ms for more precise detection

    # Register error/warn changed callback
    def _error_warn_changed_callback(self, data):
        if data and data['error_code'] != 0:
            self.alive = False
            self.pprint(color('err={}, quit'.format(data['error_code']), fg="red"))
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)

    # Register state changed callback
    def _state_changed_callback(self, data):
        if not self._ignore_exit_state and data and data['state'] == 4:
            self.alive = False
            self.pprint(color('state=4, quit', fg="red"))
            self._arm.release_state_changed_callback(self._state_changed_callback)

        # NOTE: here's where I'd need to add the code to clear the error, if appropriate

    def _check_code(self, code, label):
        if not self.is_alive or code != 0:
            self.alive = False
            ret1 = self._arm.get_state()
            ret2 = self._arm.get_err_warn_code()
            self.pprint(color('{}, code={}, connected={}, state={}, error={}, ret1={}. ret2={}'.format(
                label, code, self._arm.connected, self._arm.state, self._arm.error_code, ret1, ret2), fg="red"))
        return self.is_alive

    @staticmethod
    def pprint(*args, **kwargs):
        try:
            stack_tuple = traceback.extract_stack(limit=2)[0]
            print(color('[{}][{}] {}'.format(
                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), 
                stack_tuple[1], 
                ' '.join(map(str, args))), fg="cyan"))
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
    def run(self, start_event=None, completion_event=None, trajectory_events=None, trajectory_syncs=None):
        try:
            if start_event:
                start_event.wait()  # Wait for start signal
                
            # Initial gripper position
            code = self._arm.set_gripper_position(self.GRIPPER_OPEN_POS, wait=True, speed=5000, auto_enable=True)
            if not self._check_code(code, 'set_gripper_position'):
                return

            # Execute each trajectory in sequence
            for i, traj_num in enumerate(self._trajectory_numbers):
                self.blink_led(traj_num)
                
                # Use play_trajectory method instead of direct playback
                if not self.play_trajectory(f'_{traj_num}{self.name}'):
                    return
                    
                # Set gripper position associated with this trajectory
                target_pos = self._trajectory_gripper_positions.get(traj_num, self.GRIPPER_CLOSE_POS)  # Default to closed if not specified
                code = self._arm.set_gripper_position(target_pos, wait=True, speed=5000, auto_enable=True)
                if not self._check_code(code, 'set_gripper_position'):
                    return

                if trajectory_events and i < len(trajectory_events):
                    trajectory_events[i].set()  # Signal completion of current trajectory
                
                # Wait for both arms to complete current trajectory before continuing
                if trajectory_syncs and i < len(trajectory_syncs):
                    trajectory_syncs[i].wait()

            if completion_event:
                completion_event.set()  # Signal completion

        except Exception as e:
            self.pprint(color('MainException: {}'.format(e), fg="red"))
        finally:
            self.alive = False
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
            self._arm.release_state_changed_callback(self._state_changed_callback)


if __name__ == '__main__':
    RobotMain.pprint(color('xArm-Python-SDK Version:{}'.format(version.__version__), fg="green"))

    # Define trajectory configuration
    trajectory_config = {
        'trajectories': [1, 2],  # Trajectory numbers
        'gripper_positions': [0, RobotMain.GRIPPER_OPEN_POS]  # Corresponding gripper positions
    }

    for _ in range(1):  # Execute once
        # Create synchronization events
        start_event = threading.Event()
        left_completion = threading.Event()
        right_completion = threading.Event()
        stop_monitor = threading.Event()  # Event to stop monitoring threads

        # Create events for each trajectory
        left_trajectory_events = [threading.Event() for _ in trajectory_config['trajectories']]
        right_trajectory_events = [threading.Event() for _ in trajectory_config['trajectories']]
        trajectory_syncs = [threading.Event() for _ in trajectory_config['trajectories']]

        # Initialize both arms
        arm_left = XArmAPI('192.168.1.236', baud_checkset=False)
        arm_right = XArmAPI('192.168.1.218', baud_checkset=False)
        time.sleep(0.5)
        robot_left = RobotMain(arm_left, name='_left', trajectory_config=trajectory_config)
        robot_right = RobotMain(arm_right, name='_right', trajectory_config=trajectory_config)

        # "monitoring" threads for both arms
        left_monitor = threading.Thread(target=robot_left.monitor_digital_input, args=(stop_monitor,))
        right_monitor = threading.Thread(target=robot_right.monitor_digital_input, args=(stop_monitor,))
        left_monitor.daemon = True
        right_monitor.daemon = True
        #
        left_monitor.start()
        right_monitor.start()

        #------------------------------------------------------------------------------------------------
        # "run" threads for both arms
        left_thread = threading.Thread(target=robot_left.run, args=(start_event, left_completion, left_trajectory_events, trajectory_syncs))
        right_thread = threading.Thread(target=robot_right.run, args=(start_event, right_completion, right_trajectory_events, trajectory_syncs))
        #        
        left_thread.start()
        right_thread.start()

        # Signal both arms to start
        start_event.set()

        # Wait for each trajectory to complete and signal continuation
        for i in range(len(trajectory_config['trajectories'])):
            left_trajectory_events[i].wait()
            right_trajectory_events[i].wait()
            trajectory_syncs[i].set()

        # Wait for both arms to complete
        left_completion.wait()
        right_completion.wait()

        # Wait for threads to finish
        left_thread.join()
        right_thread.join()
        #------------------------------------------------------------------------------------------------

        try:
            while True:
                time.sleep(1)  # Small sleep to prevent busy waiting
        except KeyboardInterrupt:
            print(color("\nCtrl+C detected, stopping robots...", fg="yellow"))


        # Signal monitor threads to stop and wait briefly
        stop_monitor.set()
        time.sleep(0.1)

        # Clean up resources before next iteration
        arm_left.disconnect()
        arm_right.disconnect()
        time.sleep(1)  # Brief pause between iterations
