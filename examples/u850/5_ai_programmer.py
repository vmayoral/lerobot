import time
import math
import sys
import queue
import datetime
import random
import traceback
import threading
from xarm import version
from xarm.wrapper import XArmAPI
from wasabi import color
import whisper
import os
import pyaudio
import wave
import numpy as np
import tempfile
import pvporcupine
import struct
from openai import OpenAI
import re
from typing import List, Dict, Any, Literal, Optional, Union
from typing_extensions import Annotated
import json
from functools import wraps
from typing import Callable, TypeVar, ParamSpec, cast
import inspect

P = ParamSpec('P')
T = TypeVar('T')

def llm_enabled(description: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to mark methods as LLM-enabled and provide a high-level description
    
    Args:
        description: High-level description of what this method does, used by LLM
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        func.llm_enabled = True  # type: ignore
        func.llm_description = description  # type: ignore
        return func
    return decorator

class RobotMain(object):
    """Robot Main Class for controlling xArm robots

    This class provides control interfaces for xArm robots with both programmatic
    and natural language (LLM) control capabilities.
    
    LLM-Enabled Methods:
        - set_manual_mode: Switch robot to manual control mode
        - set_motion_mode: Switch robot to motion mode
        - set_led: Control the LED light on the robot
        - set_gripper: Control the gripper position

    To make methods LLM-controllable:
        Methods marked with @llm_enabled decorator are available for natural language control.
        See individual method documentation for details.
    
    To make methods LLM-controllable:
    1. Add @llm_enabled decorator with high-level description
    2. Add type hints with Annotated types for parameters
    3. Add clear docstrings
    4. Use simple parameter types (bool, int, float, str, or Literal)
    
    Example:        ```python
        @llm_enabled("High level description for LLM")
        def method_name(self, param: Annotated[type, "description"]) -> None:
            '''Detailed description of what the method does
            
            Args:
                param: Detailed parameter description
            '''
            # Implementation        ```
    """
    # Gripper position constants
    GRIPPER_OPEN_POS = 800
    GRIPPER_APPROACH_POS = 200
    GRIPPER_CLOSE_POS = 0

    def __init__(self, robot, name, trajectory_config=None, **kwargs):
        self.name = name
        self.alive = True
        self._arm = robot
        self._ignore_exit_state = False
        self._tcp_speed = 100
        self._tcp_acc = 2000
        self._angle_speed = 20
        self._angle_acc = 500
        self.long_click_state = False
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

    @llm_enabled("Control the LED light on the robots. Set the LED to on or off. True or False should also be recognized as valid commands.")
    def set_led(self, state: Annotated[bool, "Whether to turn the LED on or off"]) -> None:
        """Control the LED state of the robot
        
        Args:
            state: True to turn LED on, False to turn it off
        """
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

    def stop_robot(self):
        """Stop the robot"""
        self._arm.emergency_stop()

    @llm_enabled("Clear the robot errors. This any and all errors that have occurred on the robot. Reinitialize the robot.")
    def clear_errors(self):
        """Clear the robot errors"""

        self._arm.clean_warn()
        self._arm.clean_error()
        self._arm.motion_enable(True)
        self._arm.set_gripper_enable(True)
        # Reset to motion mode
        self.set_motion_mode()
        # Reset LED
        self.set_led(False)

        # Clean up any gripper errors before finishing
        self._arm.clean_gripper_error()

        # Brief pause to let settings take effect
        time.sleep(0.1)
        self.alive = True

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

    @llm_enabled("Switch, turn or set robot arm to manual mode. Either right or left arm can be specified as well, including both.")
    def set_manual_mode(self, max_retries: Annotated[Optional[int], "Maximum number of retries before failing"]=5) -> None:
        """Set the robot arm to manual mode (mode 2) with error handling and retries
        
        Args:
            max_retries: Maximum number of times to retry setting manual mode before failing. Defaults to 5.
        """
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

    @llm_enabled("Switch or set robot arm to motion mode. Either right or left arm can be specified as well, including both.")
    def set_motion_mode(self) -> None:
        """Set the robot arm to motion mode (mode 0) for automated trajectory execution
        
        This mode is used for playing back recorded trajectories and automated movements.
        The robot will automatically clean errors and set the appropriate state.
        """
        self._arm.set_mode(0)
        self._arm.set_state(0)

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
            
        # extract the trajectory number from the name
        traj_num = int(trajectory_name.strip('_').split('_')[0])
        time.sleep(1.5)
        self.blink_led(traj_num)

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

    @llm_enabled("Add a new trajectory. This will start recording a new trajectory. Optionally specify a trajectory number.")
    def add_new_trajectory(self, trajectory_number: Annotated[Optional[int], "The trajectory number to use. If not provided, will use next available number"] = None) -> str:
        """Add a new trajectory by starting to record it and return the trajectory name.

        Args:
            trajectory_number: Optional trajectory number to use (1-99). If not provided,
                          will automatically use the next available number.
        Returns:
            str: Name of the new trajectory in format '_<number><arm>'
        Raises:
            ValueError: If trajectory_number is invalid or already exists

        NOTE: stop and save should happen outside
        """
        if self._current_recording is not None:
            return self._current_recording  # recording already in progress

        # Validate trajectory_number if provided
        if trajectory_number is not None:
            if not isinstance(trajectory_number, int):
                raise ValueError("Trajectory number must be an integer")
        else:
            trajectory_number = max(self._trajectory_numbers) + 1 if self._trajectory_numbers else 1

        trajectory_name = f'_{trajectory_number}{self.name}'
        print(color(f"Started recording new trajectory: {trajectory_name}", fg="green"))
        self.set_manual_mode()
        self.start_recording_trajectory(trajectory_name)

        # Handle long click state for manual button control
        if not self.long_click_state:
            self.long_click_state = True
            self.set_led(True)  # mark that we're in long click mode
            self._current_recording = trajectory_name

        return trajectory_name

    def monitor_digital_input(self, stop_event):
        single_click_time = 0.2
        double_click_time = 0.5
        long_click_time = 1.0

        last_press_time = 0
        last_click_time = 0
        long_click_detected = False
        click_count = 0        

        while not stop_event.is_set():
            code, value = self._arm.get_tgpio_digital(ionum=2)
            if code == 0:
                current_time = time.time()

                if value == 1:  # Button pressed
                    if last_press_time == 0:
                        last_press_time = current_time
                    elif not long_click_detected and current_time - last_press_time >= long_click_time:
                        long_click_detected = True
                        self.long_click_state = not self.long_click_state
                        if self.long_click_state:
                            self.set_led(True)
                            try:
                                self._current_recording = self.add_new_trajectory()
                                print(color(f"Long click detected -> Started recording a new trajectory: {self._current_recording}", fg="blue"))
                            except RuntimeError as e:
                                print(color(f"Warning: {str(e)}", fg="yellow"))
                        else:
                            self.set_led(False)
                            try:
                                print(color(f"Long click detected -> Stopped recording the current trajectory: {self._current_recording}", fg="blue"))
                                self.stop_recording_trajectory(self._current_recording)
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
            
            self.clear_errors()
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

    @llm_enabled("Set the robot gripper. Control the robot gripper position, also referred to as the robot hand. Open or close the left/right gripper should also be recognized as valid commands.")
    def set_gripper(self, position: Annotated[Union[Literal["open", "close"], int], 
                                           "Position can be 'open', 'close', or a specific position (0-800)"]) -> None:
        """Control the robot gripper position
        
        Args:
            position: Can be 'open', 'close', or a specific position value between 50-800.
                     'open' sets to fully open position (800)
                     'close' sets to fully closed position (50)
                     Numeric values should be between 50-800
        
        Raises:
            ValueError: If position value is outside valid range or invalid input
            RuntimeError: If gripper control fails
        """
        # Convert string commands to positions
        if isinstance(position, str):
            position_lower = position.lower()
            if position_lower == "open":
                target_pos = self.GRIPPER_OPEN_POS
            elif position_lower == "closed" or position_lower == "close":
                target_pos = self.GRIPPER_CLOSE_POS
            else:
                # Try to convert string to number
                try:
                    target_pos = int(position)
                except ValueError:
                    raise ValueError(
                        f"Invalid position string: {position}. Must be 'open', 'close[d]', or a number between "
                        f"{self.GRIPPER_CLOSE_POS} and {self.GRIPPER_OPEN_POS}"
                    )
        else:
            target_pos = position

        # Validate numeric position
        if not (self.GRIPPER_CLOSE_POS <= target_pos <= self.GRIPPER_OPEN_POS):
            raise ValueError(
                f"Position must be between {self.GRIPPER_CLOSE_POS} and {self.GRIPPER_OPEN_POS}"
            )

        # Execute gripper movement
        code = self._arm.set_gripper_position(
            pos=target_pos,
            wait=True,
            speed=5000,
            auto_enable=True
        )
        
        if not self._check_code(code, 'set_gripper_position'):
            raise RuntimeError(f"Failed to set gripper position to {position}")

    # Robot Main Run
    def run(self, start_event=None, completion_event=None, trajectory_events=None, trajectory_syncs=None, trajectory_config=None):
        try:
            if start_event:
                start_event.wait()  # Wait for start signal
                
            try:
                # Initial gripper position
                self.set_gripper(RobotMain.GRIPPER_APPROACH_POS)
            except Exception as e:
                self.pprint(color(f'Failed to open gripper: {e}', fg="red"))
                return

            # Execute each trajectory in sequence
            for i, traj_num in enumerate(trajectory_config['trajectories']):
                self.blink_led(traj_num)
                
                # Use play_trajectory method instead of direct playback
                if not self.play_trajectory(f'_{traj_num}{self.name}'):
                    return
                    
                # Set gripper position associated with this trajectory
                target_pos = self._trajectory_gripper_positions.get(traj_num, self.GRIPPER_CLOSE_POS)
                try:
                    self.set_gripper(target_pos)
                except Exception as e:
                    self.pprint(color(f'Failed to set gripper position: {e}', fg="red"))
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
            # self.alive = False
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
            self._arm.release_state_changed_callback(self._state_changed_callback)



    @classmethod
    def get_function_definitions(cls) -> list:
        """Generate OpenAI function definitions from LLM-enabled methods"""
        import inspect
        from typing import get_type_hints, get_args, get_origin, Union, List
        
        function_definitions = []
        
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            # Skip methods not marked as LLM-enabled
            if not hasattr(method, 'llm_enabled'):
                continue
                
            # Get signature and docstring
            sig = inspect.signature(method)
            doc = inspect.getdoc(method)
            type_hints = get_type_hints(method, include_extras=True)
            
            parameters = {
                "type": "object",
                "properties": {
                    "arm": {
                        "type": "string",
                        "enum": ["left", "right", "both"],
                        "description": "Which arm to control: left, right, or both",
                        "default": "both"
                    }
                },
                "required": []
            }
            
            # Process each parameter
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                    
                param_type = type_hints.get(param_name)
                if not param_type:
                    continue

                # Debug print
                print(f"Processing parameter {param_name} with type {param_type}")
                    
                # Initialize parameter schema
                param_schema = {}
                description = None
                is_optional = False
                base_type = param_type
                
                # Handle Optional types
                if get_origin(param_type) == Union and type(None) in get_args(param_type):
                    is_optional = True
                    base_type = next(t for t in get_args(param_type) if t != type(None))
                    
                # Handle Annotated types
                if get_origin(base_type) == Annotated:
                    annotated_args = get_args(base_type)
                    base_type = annotated_args[0]
                    if len(annotated_args) > 1:
                        description = annotated_args[1]
                
                # Handle List types
                if get_origin(base_type) == list or get_origin(base_type) == List:
                    param_schema["type"] = "array"
                    list_type = get_args(base_type)[0]
                    if list_type == int:
                        param_schema["items"] = {"type": "integer"}
                    elif list_type == str:
                        param_schema["items"] = {"type": "string"}
                    elif list_type == float:
                        param_schema["items"] = {"type": "number"}
                    elif list_type == bool:
                        param_schema["items"] = {"type": "boolean"}
                # Handle basic types
                elif base_type == bool:
                    param_schema["type"] = "boolean"
                elif base_type == int:
                    param_schema["type"] = "integer"
                elif base_type == float:
                    param_schema["type"] = "number"
                elif base_type == str:
                    param_schema["type"] = "string"
                elif get_origin(base_type) == Literal:
                    param_schema["type"] = "string"
                    param_schema["enum"] = list(get_args(base_type))
                
                # Add description if available
                if description:
                    param_schema["description"] = description
                    
                # Add parameter to properties if we successfully determined its type
                if param_schema:
                    parameters["properties"][param_name] = param_schema
                    # Add to required list if not optional and no default value
                    if not is_optional and param.default == param.empty:
                        parameters["required"].append(param_name)
            
            function_def = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": getattr(method, 'llm_description', doc.split('\n')[0] if doc else ""),
                    "parameters": parameters
                }
            }
            
            # # Debug print
            # print(f"\nGenerated function definition for {name}:")
            # print(json.dumps(function_def, indent=2))
            
            function_definitions.append(function_def)
            
        return function_definitions

    @llm_enabled("Run or execute trajectories on the robot arm. Could be on the left or right arm, or both.")
    def execute_trajectories(self, 
                            trajectory_indices: Annotated[List[int], "List of trajectory indices to execute (e.g. [0,1,2]). If not specified, executes all trajectories in order"]=None,
                            wait_for_start: Annotated[bool, "Whether to wait for a start signal before beginning"]=False,
                            synchronize_arms: Annotated[bool, "Whether to synchronize movements between left and right arms"]=True) -> None:
        """Execute the robot's sequence of recorded trajectories
        
        Args:
            trajectory_indices: List of trajectory indices (0-based) to execute. If None, executes all trajectories
            wait_for_start: If True, wait for an external start signal before beginning
            synchronize_arms: If True, synchronize movements between left and right arms

        NOTE: This function is not used and sits here as a placeholder because it does not have access
        to both robots for synchronization. What's used is the execute_trajectories_sync function below.
        """
        pass

class FunctionCallParser:
    """Parser for different LLM function call formats"""
    
    @staticmethod
    def parse_function_calls_qwen(content: str) -> List[Dict[str, Any]]:
        """Parse function calls from Qwen-like model responses"""
        function_calls = []
        
        # Try to parse tool_call blocks first
        for m in re.finditer(r"<tool_call>\n(.+?)\n</tool_call>", content, re.DOTALL):
            try:
                # First try parsing as proper JSON
                try:
                    func = json.loads(m.group(1))
                except json.JSONDecodeError:
                    # If that fails, try to fix unquoted property names
                    fixed_json = re.sub(
                        r'(?<![\w"])([\w]+)(?=\s*:)', 
                        r'"\1"', 
                        m.group(1)
                    )
                    func = json.loads(fixed_json)
                
                function_name = func.get("name", "")
                function_args = func.get("arguments", {})
                if isinstance(function_args, str):
                    function_args = json.loads(function_args)
                
                function_calls.append({
                    "name": function_name,
                    "arguments": function_args
                })
            except json.JSONDecodeError as e:
                print(color(f"Failed to parse tool call: {m.group(1)}. Error: {e}", fg="red"))

        # If no tool_call blocks found, try to parse the entire content
        if not function_calls:
            try:
                # Look for function call pattern in content
                match = re.search(r"function:\s*(\w+).*arguments:\s*({[^}]+})", content, re.IGNORECASE | re.DOTALL)
                if match:
                    name = match.group(1)
                    # Fix unquoted property names in arguments JSON
                    args_json = re.sub(
                        r'(?<![\w"])([\w]+)(?=\s*:)', 
                        r'"\1"', 
                        match.group(2)
                    )
                    args = json.loads(args_json)
                    function_calls.append({
                        "name": name,
                        "arguments": args
                    })
            except Exception as e:
                print(color(f"Failed to parse function call from content: {e}", fg="red"))

        return function_calls

    @staticmethod
    def parse_function_calls_llama(content: str) -> List[Dict[str, Any]]:
        """Parse function calls from Llama-like model responses"""
        pattern = r"<function=([a-zA-Z0-9_]+)>\s*(\{.*?\})\s*</function>"
        matches = re.findall(pattern, content, re.DOTALL)
        function_calls = []

        for function_name, args_string in matches:
            function_name = function_name.strip()
            args_string = args_string.strip()

            if not function_name:
                continue  # Skip if function name is empty

            try:
                # Attempt to parse the arguments as JSON
                args = json.loads(args_string)
            except json.JSONDecodeError:
                # Fallback to custom parsing if JSON parsing fails
                args = FunctionCallParser.custom_parse(args_string)

            function_calls.append({"name": function_name, "arguments": args})

        return function_calls

    @staticmethod
    def custom_parse(args_string: str) -> Dict[str, Any]:
        """Custom parsing method for when JSON parsing fails"""
        args = {}
        # Remove any surrounding whitespace and curly braces
        cleaned_args = args_string.strip().strip("{}")

        # Use a regex to find key-value pairs, supporting nested JSON objects or arrays
        pair_pattern = r'"?(\w+)"?\s*:\s*(?:"([^"\\]*(?:\\.[^"\\]*)*)"|(\d+\.\d+|\d+)|(\w+)|(\{.*?\})|(\[.*?\]))'
        matches = re.findall(pair_pattern, cleaned_args)

        for match in matches:
            key = match[0]
            value = None
            if match[1]:  # String value
                # Handle escaped characters
                value = bytes(match[1], "utf-8").decode("unicode_escape")
            elif match[2]:  # Number (int or float)
                num_str = match[2]
                try:
                    if '.' in num_str:
                        value = float(num_str)
                    else:
                        value = int(num_str)
                except ValueError:
                    value = num_str
            elif match[3]:  # Boolean or None
                bool_str = match[3].lower()
                if bool_str == "true":
                    value = True
                elif bool_str == "false":
                    value = False
                elif bool_str == "none":
                    value = None
                else:
                    value = bool_str
            elif match[4]:  # Nested JSON object
                try:
                    value = json.loads(match[4])
                except json.JSONDecodeError:
                    value = match[4]
            elif match[5]:  # Array
                try:
                    value = json.loads(match[5])
                except json.JSONDecodeError:
                    value = match[5]
            else:
                value = match[0]  # Fallback to key as value

            args[key] = value

        return args

    @staticmethod
    def parse_function_calls(content: str, model_type: str) -> List[Dict[str, Any]]:
        """Parse function calls based on model type
        
        Args:
            content: The model response content
            model_type: Type of model ('qwen', 'llama', etc.)
        """
        if 'qwen' in model_type.lower():
            return FunctionCallParser.parse_function_calls_qwen(content)
        elif any(x in model_type.lower() for x in ['llama', 'mistral', 'phi', 'watt-tool']):
            return FunctionCallParser.parse_function_calls_llama(content)
        else:
            # Default to qwen parser
            print(color(f"Warning: Unknown model type {model_type}, defaulting to qwen parser", fg="yellow"))
            return FunctionCallParser.parse_function_calls_qwen(content)

##############################################################################################################
#                                           FUNCTIONS
##############################################################################################################

def run_robot_threads(robot_left, robot_right, start_event, left_completion, right_completion, 
                     left_trajectory_events, right_trajectory_events, trajectory_syncs, trajectory_config):
    """Set up and manage robot execution threads"""

    print(trajectory_config)

    # Start robot threads
    left_thread = threading.Thread(target=robot_left.run, 
                                 args=(start_event, left_completion, left_trajectory_events, trajectory_syncs, trajectory_config))
    right_thread = threading.Thread(target=robot_right.run, 
                                  args=(start_event, right_completion, right_trajectory_events, trajectory_syncs, trajectory_config))
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

def setup_monitoring_threads(robot_left, robot_right, stop_monitor):
    """Set up and start monitoring threads for both robot arms"""
    left_monitor = threading.Thread(target=robot_left.monitor_digital_input, args=(stop_monitor,))
    right_monitor = threading.Thread(target=robot_right.monitor_digital_input, args=(stop_monitor,))
    left_monitor.daemon = True
    right_monitor.daemon = True
    left_monitor.start()
    right_monitor.start()
    return left_monitor, right_monitor

def setup_wake_word_thread(stop_monitor, speech_trigger, mic_lock, robot_left, robot_right):
    """Set up wake word detection thread using Porcupine"""
    
    def wake_word_loop(stop_event, speech_event, mic_lock):
        # Track which wake word was detected
        detected_word = None
        while not stop_event.is_set():
            porcupine = None
            pa = None
            audio_stream = None
            
            try:
                # Initialize Porcupine
                keywords = ['alexa', 'computer', 'terminator']
                porcupine = pvporcupine.create(
                    access_key=os.getenv('PORCUPINE_ACCESS_KEY'),
                    keywords=keywords
                )

                with mic_lock:
                    pa = pyaudio.PyAudio()
                    audio_stream = pa.open(
                        rate=porcupine.sample_rate,
                        channels=1,
                        format=pyaudio.paInt16,
                        input=True,
                        frames_per_buffer=porcupine.frame_length
                    )

                    print(color("Wake word detection started... 👂", fg="yellow"))

                    while not stop_event.is_set():
                        try:
                            pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
                            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

                            keyword_index = porcupine.process(pcm)
                            if keyword_index >= 0:
                                detected_word = keywords[keyword_index]
                                print(color(f"Wake word '{detected_word}' detected! 🎯", fg="green"))
                                
                                # Handle "terminator" wake word
                                if detected_word == 'terminator':
                                    print(color("TERMINATOR detected! Stopping both robots! 🛑", fg="red", bold=True))
                                    robot_left.stop_robot()
                                    robot_right.stop_robot()
                                else:
                                    # Release microphone lock before triggering speech recognition
                                    audio_stream.stop_stream()
                                    audio_stream.close()
                                    pa.terminate()
                                    # Pass detected word through speech_event
                                    speech_event.detected_word = detected_word
                                    speech_event.set()  # Trigger speech recognition
                                    time.sleep(0.5)  # Brief pause before listening again
                                    speech_event.clear()
                                    break  # Exit inner loop to reinitialize audio

                        except IOError as e:
                            print(color(f"Audio stream error: {str(e)}", fg="yellow"))
                            break  # Break inner loop to recreate stream

            except Exception as e:
                print(color(f"Wake word detection error: {str(e)}", fg="red"))
                time.sleep(1)  # Sleep before retrying

            finally:
                # Clean up resources before next iteration
                if audio_stream is not None:
                    try:
                        audio_stream.stop_stream()
                        audio_stream.close()
                    except:
                        pass
                if pa is not None:
                    try:
                        pa.terminate()
                    except:
                        pass
                if porcupine is not None:
                    try:
                        porcupine.delete()
                    except:
                        pass
                
            # Small delay before recreating resources
            time.sleep(0.5)

    wake_thread = threading.Thread(target=wake_word_loop, args=(stop_monitor, speech_trigger, mic_lock))
    wake_thread.daemon = True
    wake_thread.start()
    return wake_thread

def setup_speech_thread(stop_monitor, speech_trigger, mic_lock, robot_left, robot_right):
    """Set up and start speech recognition thread"""

    def play_beep(freq=440, duration=0.1):
        """Play a beep sound through system audio
        Args:
            freq (int): Frequency in Hz (default: 880 Hz = A5 note). Common values:
                        440 Hz = A4 note
                        523 Hz = C5 note 
                        587 Hz = D5 note
                        659 Hz = E5 note
                        698 Hz = F5 note
                        784 Hz = G5 note
                        880 Hz = A5 note
                        932 Hz = Bb5 note
                        1047 Hz = C6 note
            duration (float): Duration in seconds (default: 0.1s)
        """
        beep_audio = pyaudio.PyAudio()
        beep_stream = beep_audio.open(format=pyaudio.paFloat32,
                                    channels=1,
                                    rate=44100,
                                    output=True)
        samples = (np.sin(2*np.pi*np.arange(44100*duration)*freq/44100)).astype(np.float32)
        beep_stream.write(samples.tobytes())
        beep_stream.stop_stream()
        beep_stream.close()
        beep_audio.terminate()

    def generate_function_help_text(function_calls):
        """Generate a human-readable description of available functions"""
        help_text = "Available functions:\n"
        
        for func in function_calls:
            if func["type"] != "function":
                continue
                
            f = func["function"]
            help_text += f"\n- {f['name']}: {f['description']}\n"
            
            # Add parameter descriptions
            params = f["parameters"].get("properties", {})
            if params:
                help_text += "  Parameters:\n"
                for param_name, param_info in params.items():
                    desc = param_info.get("description", "")
                    param_type = param_info.get("type", "")
                    if "enum" in param_info:
                        param_type = f"{param_type} (one of: {', '.join(map(str, param_info['enum']))})"
                    help_text += f"    - {param_name} ({param_type}): {desc}\n"
            
        return help_text

    def execute_function(robot_left, robot_right, call):
        """Execute a function call on specified robot(s)"""
        # Extract and remove arm selection from arguments
        args = call['arguments'].copy()
        target_arm = args.pop('arm', 'both').lower()
        
        # Special handling for execute_trajectories
        if call['name'] == 'execute_trajectories':
            # Extract trajectory_indices if provided
            trajectory_indices = args.get('trajectory_indices')
            if trajectory_indices:
                # Convert string representation of list to actual list if needed
                if isinstance(trajectory_indices, str):
                    try:
                        # Handle various string formats like "[1,2]" or "1, 2"
                        trajectory_indices = json.loads(trajectory_indices.replace(' ', ''))
                    except json.JSONDecodeError:
                        # Try comma-separated format
                        trajectory_indices = [int(x.strip()) for x in trajectory_indices.strip('[]').split(',')]
                
                # Ensure trajectory_indices is a list of integers
                if not isinstance(trajectory_indices, list):
                    trajectory_indices = [trajectory_indices]
                trajectory_indices = [int(idx) for idx in trajectory_indices]
                
                # Validate indices are within bounds
                #
                # Note: start from 1 because the trajectory numbers start from 1
                max_index = len(robot_left._trajectory_numbers)
                if not all(1 <= idx <= max_index for idx in trajectory_indices):
                    raise ValueError(f"Trajectory indices must be between 1 and {max_index}")
                    
                # Create new temporary trajectory config with only selected trajectories
                #
                # Note: We subtract 1 from the indices because the trajectory numbers start from 1
                selected_trajectories = [robot_left._trajectory_numbers[idx - 1] for idx in trajectory_indices]
                selected_gripper_positions = [robot_left._gripper_positions[idx - 1] for idx in trajectory_indices]
                
                temp_trajectory_config = {
                    'trajectories': selected_trajectories,
                    'gripper_positions': selected_gripper_positions
                }
            else:
                # If no indices provided, use all trajectories
                temp_trajectory_config = {
                    'trajectories': robot_left._trajectory_numbers[:],
                    'gripper_positions': robot_left._gripper_positions[:]
                }

            # Create events for selected trajectories only
            start_event = threading.Event()
            left_completion = threading.Event()
            right_completion = threading.Event()
            left_trajectory_events = [threading.Event() for _ in temp_trajectory_config['trajectories']]
            right_trajectory_events = [threading.Event() for _ in temp_trajectory_config['trajectories']]
            trajectory_syncs = [threading.Event() for _ in temp_trajectory_config['trajectories']] if args.get('synchronize_arms', True) else None

            # Run the trajectories using the temporary config
            run_robot_threads(
                robot_left=robot_left,
                robot_right=robot_right,
                start_event=start_event,
                left_completion=left_completion,
                right_completion=right_completion,
                left_trajectory_events=left_trajectory_events,
                right_trajectory_events=right_trajectory_events,
                trajectory_syncs=trajectory_syncs,
                trajectory_config=temp_trajectory_config  # Use temporary config with selected trajectories
            )
            return True

        # Normal function execution for other functions
        robots_to_use = []
        if target_arm in ('both', 'left'):
            robots_to_use.append(('left', robot_left))
        if target_arm in ('both', 'right'):
            robots_to_use.append(('right', robot_right))
        
        # Validate function exists on robots
        if not all(hasattr(robot, call['name']) for _, robot in robots_to_use):
            print(color(f"Error: Function {call['name']} not found on specified robot(s)", fg="red"))
            return False
        
        success = True
        for arm_name, robot in robots_to_use:
            try:
                # Get the method
                method = getattr(robot, call['name'])
                
                # Validate arguments against function signature
                sig = inspect.signature(method)
                try:
                    # This will raise TypeError if arguments don't match
                    sig.bind(**args)
                    
                    # If validation passes, execute
                    method(**args)
                    print(color(f"Successfully executed {call['name']} on {arm_name} arm", fg="green"))
                except TypeError as e:
                    print(color(f"Invalid arguments for {call['name']} on {arm_name} arm: {str(e)}", fg="red"))
                    success = False
                    
            except Exception as e:
                print(color(f"Error executing {call['name']} on {arm_name} arm: {str(e)}", fg="red"))
                success = False
        
        return success

    def speech_recognition_loop(stop_event, trigger_event, mic_lock):
        # Initialize whisper model outside audio context
        # model = whisper.load_model("turbo")
        model = whisper.load_model("tiny")

        # Initialize LLM client
        llm_client = OpenAI(
            base_url="http://localhost:8000/v1", 
            api_key="1234"
        )

        # Get function definitions and generate help text
        function_calls = RobotMain.get_function_definitions()
        function_help = generate_function_help_text(function_calls)
        
        print(color("\nFunction calls:", fg="blue", bold=True))
        for func in function_calls:
            f = func['function']
            print(color(f"\n  {f['name']}:", fg="cyan", bold=True))
            print(color(f"    Description: {f['description']}", fg="green"))
            
            # Print parameters if they exist
            if 'parameters' in f and 'properties' in f['parameters']:
                print(color("    Parameters:", fg="yellow", bold=True))
                for param_name, param in f['parameters']['properties'].items():
                    desc = param.get('description', 'No description')
                    if 'enum' in param:
                        desc += f" (Valid values: {', '.join(map(str, param['enum']))})"
                    print(color(f"      - {param_name}: ", fg="blue") + color(desc, fg="white"))
        print()

        while not stop_event.is_set():
            # Wait for wake word trigger
            trigger_event.wait()

            # Set recording duration based on detected wake word
            RECORD_SECONDS = 12 if trigger_event.detected_word == 'alexa' else 5

            p = None
            stream = None
            try:
                with mic_lock:
                    p = pyaudio.PyAudio()

                    # Audio recording parameters optimized for Whisper
                    # Whisper expects 16kHz sample rate and 16-bit PCM audio
                    CHUNK = 4096  # Increased for better buffering
                    FORMAT = pyaudio.paInt16  # Changed to 16-bit PCM which Whisper prefers
                    CHANNELS = 1  # Mono audio is sufficient for speech recognition
                    RATE = 16000  # Whisper's expected sample rate

                    play_beep(freq=440, duration=0.1)  # Play beep before recording

                    # Open audio stream
                    stream = p.open(format=FORMAT,
                                    channels=CHANNELS, 
                                    rate=RATE,
                                    input=True,
                                    frames_per_buffer=CHUNK)

                    try:
                        # Record audio
                        print(color(f"Starting speech recognition for {RECORD_SECONDS} seconds... 🎤 🗣️ 👂", fg="green"))
                        frames = []
                        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                            if stop_event.is_set():
                                break
                            data = stream.read(CHUNK, exception_on_overflow=False)
                            frames.append(data)

                        if stop_event.is_set():
                            break

                        # Convert frames to numpy array
                        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)

                        # Release microphone before processing
                        stream.stop_stream()
                        stream.close()
                        p.terminate()

                        # Create temporary WAV file
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                            wav_path = temp_wav.name
                            with wave.open(wav_path, 'wb') as wf:
                                wf.setnchannels(CHANNELS)
                                wf.setsampwidth(p.get_sample_size(FORMAT))
                                wf.setframerate(RATE)
                                wf.writeframes(audio_data.tobytes())

                            # Play a beep to signal recording completion
                            play_beep(freq=440, duration=0.3)

                            # Transcribe audio
                            result = model.transcribe(wav_path, language="en")
                            transcribed_text = result["text"].strip()
                            if transcribed_text:  # Only process if there's actual speech
                                print(color("Speech Recognition: " + transcribed_text, fg="blue"))

                                # Prepare messages for LLM with enhanced system prompt
                                messages = [
                                    {
                                        "role": "system",
                                        "content": (
                                            "You are a robot control assistant that interprets natural language commands "
                                            "and maps them to specific function calls concerning actions on robot arms. You should only use the available "
                                            "functions and their exact parameter names as defined below.\n\n"
                                            f"{function_help}\n\n"
                                            "Rules:\n"
                                            "1. Only use functions exactly as specified above\n"
                                            "2. If a command doesn't match any available function, explain why\n"
                                            "3. Use exact parameter names as shown\n"
                                            "4. If a parameter is optional and not specified in the command, don't include it\n"
                                            "5. All JSON properties must be enclosed in double quotes\n"
                                            "6. Respond with function calls in the format:\n"
                                            "<tool_call>\n"
                                            '{"name": "function_name", "arguments": {"param": "value"}}\n'
                                            "</tool_call>"
                                        )
                                    },
                                    {
                                        "role": "user", 
                                        "content": (
                                            f"Based on the available functions, what function call(s) should be made for "
                                            f"this command: '{transcribed_text}'"
                                        )
                                    }
                                ]

                                # # Get LLM response with auto-generated function definitions
                                # llm_model = "hf.co/legionarius/watt-tool-8B-GGUF:latest"
                                # response = llm_client.chat.completions.create(
                                #     model=llm_model,
                                #     messages=messages,
                                #     temperature=0.0,
                                #     stream=True,
                                # )

                                llm_model = "qwen2.5:72b"
                                # llm_model = "qwen2.5:14b"
                                response = llm_client.chat.completions.create(
                                    model=llm_model,
                                    messages=messages,
                                    temperature=0.0,
                                    stream=True,
                                    stream_options={"include_usage": True},
                                    tools=function_calls
                                )

                                # print(color(f"Function calls: {function_calls}", bg="white"))

                                # Process response and execute functions
                                content = ""
                                for chunk in response:
                                    if chunk.choices[0].delta.content:
                                        content += chunk.choices[0].delta.content
                                        if chunk.choices[0].delta.content.strip():
                                            print(color(chunk.choices[0].delta.content, fg="grey"), end="", flush=True)
                                
                                # Parse function calls using appropriate parser based on model
                                parsed_function_calls = FunctionCallParser.parse_function_calls(content, llm_model)
                                if parsed_function_calls:
                                    for call in parsed_function_calls:
                                        play_beep(freq=880, duration=0.1)
                                        print(color(f"\nExecuting function: {call['name']} with args: {call['arguments']}", fg="green"))
                                        
                                        if execute_function(robot_left, robot_right, call):
                                            print(color(f"Successfully executed {call['name']}", fg="green"))
                                        else:
                                            print(color(f"Failed to execute {call['name']}", fg="red"))
                                else:
                                    play_beep(freq=220, duration=0.1)
                                    print(color("\nNo valid function calls found in LLM response", fg="red"))

                            # Clean up temp file
                            try:
                                os.remove(wav_path)
                            except Exception as e:
                                print(color(f"Error removing temp file: {str(e)}", fg="red"))

                    except Exception as e:
                        print(color(f"Speech recognition error: {str(e)}", fg="red"))
                        time.sleep(1)
                        continue

            except Exception as e:
                print(color(f"Fatal speech recognition error: {str(e)}", fg="red"))

            finally:
                # Cleanup
                if stream is not None:
                    try:
                        stream.stop_stream()
                        stream.close()
                    except Exception as e:
                        print(color(f"Error closing stream: {str(e)}", fg="red"))
                
                if p is not None:
                    try:
                        p.terminate()
                    except Exception as e:
                        print(color(f"Error terminating PyAudio: {str(e)}", fg="red"))

    speech_thread = threading.Thread(target=speech_recognition_loop, args=(stop_monitor, speech_trigger, mic_lock))
    speech_thread.daemon = True
    speech_thread.start()
    return speech_thread

##############################################################################################################

if __name__ == '__main__':
    RobotMain.pprint(color('xArm-Python-SDK Version:{}'.format(version.__version__), fg="green"))
    # Define initial trajectory configuration
    trajectory_config = {
        'trajectories': [1, 2],  # Trajectory numbers
        'gripper_positions': [RobotMain.GRIPPER_APPROACH_POS, RobotMain.GRIPPER_APPROACH_POS]  # Corresponding gripper positions
    }

    # trajectory_config = {
    #     'trajectories': [],  # Trajectory numbers
    #     'gripper_positions': []  # Corresponding gripper positions
    # }


    # Create synchronization events
    stop_monitor = threading.Event()  # Event to stop monitoring threads
    speech_trigger = threading.Event()  # Event to trigger speech recognition
    speech_trigger.detected_word = None  # Add attribute to track detected wake word
    mic_lock = threading.Lock()  # Lock for microphone access

    # Initialize both arms
    arm_left = XArmAPI('192.168.1.236', baud_checkset=False)
    arm_right = XArmAPI('192.168.1.218', baud_checkset=False)
    time.sleep(0.5)
    robot_left = RobotMain(arm_left, name='_left', trajectory_config=trajectory_config)
    robot_right = RobotMain(arm_right, name='_right', trajectory_config=trajectory_config)

    # Start all threads
    left_monitor, right_monitor = setup_monitoring_threads(robot_left, robot_right, stop_monitor)
    wake_word_thread = setup_wake_word_thread(stop_monitor, speech_trigger, mic_lock, robot_left, robot_right)
    speech_thread = setup_speech_thread(stop_monitor, speech_trigger, mic_lock, robot_left, robot_right)
    
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