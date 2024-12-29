########################################################################################
# Utilities
########################################################################################


import logging
import time
import traceback
from contextlib import nullcontext
from copy import copy
from functools import cache

import cv2
import torch
import tqdm
from termcolor import colored
from flask import Flask, Response, send_from_directory, redirect, request
import threading
import numpy as np
from threading import Lock

from lerobot.common.datasets.populate_dataset import add_frame, safe_stop_image_writer
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.utils import get_safe_torch_device, init_hydra_config, set_global_seed
from lerobot.scripts.eval import get_pretrained_policy_path


def log_control_info(robot: Robot, dt_s, episode_index=None, frame_index=None, fps=None):
    log_items = []
    if episode_index is not None:
        log_items.append(f"ep:{episode_index}")
    if frame_index is not None:
        log_items.append(f"frame:{frame_index}")

    def log_dt(shortname, dt_val_s, color='white'):
        nonlocal log_items, fps
        info_str = f"{shortname}:{dt_val_s * 1000:5.2f} (\033[1m{1/ dt_val_s:3.1f}hz\033[0m)"
        if fps is not None:
            actual_fps = 1 / dt_val_s
            if actual_fps < fps - 1:
                info_str = colored(info_str, "yellow")
            else:
                info_str = colored(info_str, color)
        else:
            info_str = colored(info_str, color)
        log_items.append(info_str)

    # total step time displayed in milliseconds and its frequency
    log_dt("dt", dt_s, 'cyan')

    # TODO(aliberts): move robot-specific logs logic in robot.print_logs()
    if not robot.robot_type.startswith("stretch"):
        for name in robot.leader_arms:
            key = f"read_leader_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRlead", robot.logs[key], 'green')

        for name in robot.follower_arms:
            key = f"write_follower_{name}_goal_pos_dt_s"
            if key in robot.logs:
                log_dt("dtWfoll", robot.logs[key], 'blue')

            key = f"read_follower_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRfoll", robot.logs[key], 'magenta')

        for name in robot.cameras:
            key = f"read_camera_{name}_dt_s"
            if key in robot.logs:
                log_dt(f"dtR{name}", robot.logs[key], 'red')

    info_str = " ".join(log_items)
    logging.info(info_str)


@cache
def is_headless():
    """Detects if python is running without a monitor."""
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True


def has_method(_object: object, method_name: str):
    return hasattr(_object, method_name) and callable(getattr(_object, method_name))


def predict_action(observation, policy, device, use_amp):
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name in observation:
            # Check if tensor is double precision and convert if needed
            if observation[name].dtype == torch.float64:  # Explicitly check for double
                observation[name] = observation[name].double().float()  # Convert double to float32

            if "image" in name:
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)

        # Remove batch dimension
        action = action.squeeze(0)

        # Move to cpu, if not already the case
        action = action.to("cpu")

    return action


def init_keyboard_listener():
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["stop_recording"] = False

    if is_headless():
        logging.warning(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )
        listener = None
        return listener, events

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                print("Right arrow key pressed. Exiting loop...")
                events["exit_early"] = True
            elif key == keyboard.Key.left:
                print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.esc:
                print("Escape key pressed. Stopping data recording...")
                events["stop_recording"] = True
                events["exit_early"] = True
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events


def init_policy(pretrained_policy_name_or_path, policy_overrides):
    """Instantiate the policy and load fps, device and use_amp from config yaml"""
    pretrained_policy_path = get_pretrained_policy_path(pretrained_policy_name_or_path)
    hydra_cfg = init_hydra_config(pretrained_policy_path / "config.yaml", policy_overrides)
    policy = make_policy(hydra_cfg=hydra_cfg, pretrained_policy_name_or_path=pretrained_policy_path)

    # Check device is available
    device = get_safe_torch_device(hydra_cfg.device, log=True)
    use_amp = hydra_cfg.use_amp
    policy_fps = hydra_cfg.env.fps

    policy.eval()
    policy.to(device)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_global_seed(hydra_cfg.seed)
    return policy, policy_fps, device, use_amp


def warmup_record(
    robot,
    events,
    enable_teloperation,
    warmup_time_s,
    display_cameras,
    fps,
):
    control_loop(
        robot=robot,
        control_time_s=warmup_time_s,
        display_cameras=display_cameras,
        events=events,
        fps=fps,
        teleoperate=enable_teloperation,
    )


def record_episode(
    robot,
    dataset,
    events,
    episode_time_s,
    display_cameras,
    policy,
    device,
    use_amp,
    fps,
):
    control_loop(
        robot=robot,
        control_time_s=episode_time_s,
        display_cameras=display_cameras,
        dataset=dataset,
        events=events,
        policy=policy,
        device=device,
        use_amp=use_amp,
        fps=fps,
        teleoperate=policy is None,
    )


def create_camera_server(cameras_dict):
    """Creates and configures a Flask server for camera streaming"""
    app = Flask(__name__)
    frame_locks = {name: Lock() for name in cameras_dict.keys()}
    
    def generate_frames(camera_name):
        camera_key = f"observation.images.{camera_name}"
        last_frame_time = 0
        frame_interval = 1/30.0  # Cap at 30 FPS
        
        while True:
            current_time = time.time()
            if current_time - last_frame_time < frame_interval:
                time.sleep(0.001)
                continue
                
            with frame_locks[camera_key]:
                frame = cameras_dict[camera_key]
                if frame is None:
                    time.sleep(0.001)
                    continue
                
                # Convert RGB to BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Resize efficiently
                frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_NEAREST)
                
                # Use lower JPEG quality for faster transmission
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if not ret:
                    continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            last_frame_time = current_time

    @app.route('/video_feed/<camera_name>')
    def video_feed(camera_name):
        camera_key = f"observation.images.{camera_name}"
        if camera_key not in cameras_dict:
            return f"Camera not found: {camera_name}. Available cameras: {[k.split('.')[-1] for k in cameras_dict.keys()]}", 404
        return Response(generate_frames(camera_name),
                       mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/')
    def index():
        if cameras_dict:
            first_camera = list(cameras_dict.keys())[0].split('.')[-1]
            return redirect(f'/{first_camera}')
        return "No cameras available", 404

    @app.route('/<camera_name>')
    def camera_view(camera_name):
        camera_key = f"observation.images.{camera_name}"
        if camera_key not in cameras_dict:
            return f"Camera not found: {camera_name}. Available cameras: {[k.split('.')[-1] for k in cameras_dict.keys()]}", 404
            
        # Get rotation from session storage, default to 0        
        rotation = request.args.get('rotation', '0')
            
        html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>{camera_name} Camera</title>
            <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
            <style>
                body, html {{
                    margin: 0;
                    padding: 0;
                    width: 100%;
                    height: 100%;
                    overflow: hidden;
                    background: #000;
                }}
                img {{
                    width: 100vw;
                    height: 100vh;
                    object-fit: contain;
                    transition: transform 0.3s;
                }}
                .nav {{
                    position: fixed;
                    top: 10px;
                    left: 50%;
                    transform: translateX(-50%);
                    background: rgba(0,0,0,0.7);
                    padding: 10px 20px;
                    border-radius: 5px;
                    z-index: 1000;
                }}
                .nav a {{
                    color: white;
                    text-decoration: none;
                    margin: 0 15px;
                    padding: 5px 10px;
                    border-radius: 3px;
                    transition: background 0.3s;
                }}
                .nav a:hover {{
                    background: rgba(255,255,255,0.2);
                }}
                .nav a.active {{
                    background: rgba(255,255,255,0.3);
                }}
                .rotate-btn {{
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    background: rgba(0,0,0,0.7);
                    color: white;
                    border: none;
                    width: 40px;
                    height: 40px;
                    border-radius: 50%;
                    cursor: pointer;
                    z-index: 1000;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }}
                .rotate-btn:hover {{
                    background: rgba(0,0,0,0.9);
                }}
            </style>
        </head>
        <body>
            <div class="nav">
                {' '.join(f'<a href="/{cam.split(".")[-1]}" {"class=active" if cam.split(".")[-1] == camera_name else ""}>{cam.split(".")[-1]}</a>' for cam in cameras_dict.keys())}
            </div>
            <img id="camera-feed" src="/video_feed/{camera_name}?_={time.time()}" style="transform: rotate({rotation}deg)"/>
            <button class="rotate-btn" onclick="rotateImage()">⟳</button>
            
            <script>
                // Load saved rotation from localStorage or use URL param
                let currentRotation = parseInt(localStorage.getItem('{camera_name}_rotation') || '{rotation}' || '0');
                document.getElementById('camera-feed').style.transform = `rotate(${{currentRotation}}deg)`;
                
                function rotateImage() {{
                    currentRotation = (currentRotation + 90) % 360;
                    document.getElementById('camera-feed').style.transform = `rotate(${{currentRotation}}deg)`;
                    localStorage.setItem('{camera_name}_rotation', currentRotation);
                }}
            </script>
        </body>
        </html>
        '''
        return html

    return app, frame_locks


def control_loop(
    robot,
    control_time_s=None,
    teleoperate=False,
    display_cameras=False,
    dataset=None,
    events=None,
    policy=None,
    device=None,
    use_amp=None,
    fps=None,
    enable_web_server=False,
    web_server_port=5000,
):
    # TODO(rcadene): Add option to record logs
    if not robot.is_connected:
        robot.connect()

    if events is None:
        events = {"exit_early": False}

    if control_time_s is None:
        control_time_s = float("inf")

    if teleoperate and policy is not None:
        raise ValueError("When `teleoperate` is True, `policy` should be None.")

    if dataset is not None and fps is not None and dataset["fps"] != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset['fps']} != {fps}).")

    # Initialize web server if enabled
    if enable_web_server:
        observation = robot.capture_observation()
        camera_frames = {}
        frame_locks = {}
        
        # Initialize frames with first observation
        for key in observation:
            if "image" in key:
                camera_frames[key] = observation[key].numpy()
                frame_locks[key] = Lock()

        if not camera_frames:
            logging.warning("No cameras found in observation!")
            return

        logging.info(f"Found cameras: {list(camera_frames.keys())}")
        
        app, frame_locks = create_camera_server(camera_frames)
        server_thread = threading.Thread(
            target=lambda: app.run(host='0.0.0.0', port=web_server_port, debug=False, 
                                 threaded=True, use_reloader=False)
        )
        server_thread.daemon = True
        server_thread.start()
        logging.info(f"Web server started on port {web_server_port}")

    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if teleoperate:
            observation, action = robot.teleop_step(record_data=True)
        else:
            observation = robot.capture_observation()

            if policy is not None:
                pred_action = predict_action(observation, policy, device, use_amp)
                # Action can eventually be clipped using `max_relative_target`,
                # so action actually sent is saved in the dataset.
                action = robot.send_action(pred_action)
                action = {"action": action}

        if dataset is not None:
            add_frame(dataset, observation, action)

        if display_cameras and not is_headless():
            image_keys = [key for key in observation if "image" in key]
            for key in image_keys:
                cv2.imshow(key, cv2.cvtColor(observation[key].numpy(), cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        if fps is not None:
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)

        dt_s = time.perf_counter() - start_loop_t
        log_control_info(robot, dt_s, fps=fps)

        timestamp = time.perf_counter() - start_episode_t
        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Update camera frames for web server if enabled
        if enable_web_server:
            for key in observation:
                if "image" in key:
                    if key in camera_frames:
                        with frame_locks[key]:
                            camera_frames[key] = observation[key].numpy()


def reset_environment(robot, events, reset_time_s, episode_index):
    # TODO(rcadene): refactor warmup_record and reset_environment
    # TODO(alibets): allow for teleop during reset
    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()

    if has_method(robot, "robot_reset"):
        robot.robot_reset()

    timestamp = 0
    start_vencod_t = time.perf_counter()

    # Wait if necessary
    with tqdm.tqdm(total=reset_time_s, desc=f"Waiting (next episode: {episode_index + 1})") as pbar:
        while timestamp < reset_time_s:
            ## previous implementation
            ##  increments in 1 second and not controllable
            #
            # time.sleep(1)
            # timestamp = time.perf_counter() - start_vencod_t
            # pbar.update(1)

            ## current implementation
            ##  increments in teleop cycle time, controllable
            ##
            robot.teleop_step(record_data=False)
            new_timestamp = time.perf_counter() - start_vencod_t
            pbar.update(round(new_timestamp - timestamp, 2))
            timestamp = new_timestamp

            if events["exit_early"]:
                events["exit_early"] = False
                break


def stop_recording(robot, listener, display_cameras):
    robot.disconnect()

    if not is_headless():
        if listener is not None:
            listener.stop()

        if display_cameras:
            cv2.destroyAllWindows()


def sanity_check_dataset_name(repo_id, policy):
    _, dataset_name = repo_id.split("/")
    # either repo_id doesnt start with "eval_" and there is no policy
    # or repo_id starts with "eval_" and there is a policy
    if dataset_name.startswith("eval_") == (policy is None):
        raise ValueError(
            f"Your dataset name begins by 'eval_' ({dataset_name}) but no policy is provided ({policy})."
        )
