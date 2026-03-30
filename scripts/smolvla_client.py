"""
SmolVLA Remote Inference Client for SO-101.

Connects to a remote SmolVLA server (RunPod), captures camera images
and robot state, sends observations, receives action chunks, and
executes them on the SO-101 robot at 10 FPS.

Usage:
    python smolvla_client.py --server_url http://<RUNPOD_IP>:8000

Make sure to:
    1. Start the server on RunPod first (smolvla_server.py)
    2. Connect the SO-101 follower arm
    3. Set up cameras (front + wrist)
"""

import argparse
import base64
import io
import logging
import time
from collections import deque
from threading import Event, Lock, Thread

import cv2
import numpy as np
import requests

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.motors.feetech import OperatingMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def encode_image_jpeg(img_bgr: np.ndarray, quality: int = 90) -> str:
    """Encode BGR image to base64 JPEG string."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    _, jpg_buf = cv2.imencode(".jpg", img_rgb, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(jpg_buf.tobytes()).decode("ascii")


class SmolVLAClient:
    def __init__(
        self,
        server_url: str,
        robot_port: str = "/dev/ttyACM0",
        robot_id: str = "my_awesome_follower_arm",
        front_camera: str = "/dev/video0",
        wrist_camera: str = "/dev/video4",
        task: str = "Grab the brain",
        fps: float = 10.0,
        duration: float = 500.0,
        record_video: bool = False,
        video_output: str = "smolvla_eval.mp4",
        record_duration: float = 90.0,
    ):
        self.server_url = server_url.rstrip("/")
        self.task = task
        self.fps = fps
        self.duration = duration
        self.action_interval = 1.0 / fps
        self.record_video = record_video
        self.video_output = video_output
        self.record_duration = record_duration

        # Action queue
        self.action_queue = deque()
        self.queue_lock = Lock()
        self.robot_lock = Lock()
        self.shutdown_event = Event()

        # Shared frame buffer for recorder (no extra camera reads)
        self.latest_frames = {"front": None, "wrist": None}
        self.frame_lock = Lock()

        # Robot config
        self.robot_config = SO101FollowerConfig(
            port=robot_port,
            id=robot_id,
            cameras={
                "front": OpenCVCameraConfig(
                    index_or_path=front_camera, width=640, height=480, fps=30, warmup_s=3
                ),
                "wrist": OpenCVCameraConfig(
                    index_or_path=wrist_camera, width=640, height=480, fps=30, warmup_s=3
                ),
            },
        )

        # Joint names for action mapping
        self.joint_names = [
            "shoulder_pan", "shoulder_lift", "elbow_flex",
            "wrist_flex", "wrist_roll", "gripper",
        ]

    def check_server(self):
        """Check if the server is reachable."""
        try:
            resp = requests.get(f"{self.server_url}/health", timeout=5)
            data = resp.json()
            if data.get("model_loaded"):
                logger.info(f"Server is ready at {self.server_url}")
                return True
            else:
                logger.error("Server is up but model not loaded yet")
                return False
        except Exception as e:
            logger.error(f"Cannot reach server at {self.server_url}: {e}")
            return False

    def request_actions(self, obs: dict) -> list[list[float]] | None:
        """Send observation to server and get action chunk back."""
        try:
            # Encode images as JPEG base64
            # obs images are numpy arrays (H, W, 3) in BGR from OpenCV
            front_img = obs.get("front")
            wrist_img = obs.get("wrist")

            if front_img is None or wrist_img is None:
                logger.warning("Missing camera images")
                return None

            # Get robot state
            state = [obs[f"{name}.pos"] for name in self.joint_names]

            payload = {
                "camera1_jpg_b64": encode_image_jpeg(front_img),
                "camera2_jpg_b64": encode_image_jpeg(wrist_img),
                "state": state,
                "task": self.task,
            }

            t_start = time.perf_counter()
            resp = requests.post(
                f"{self.server_url}/predict",
                json=payload,
                timeout=30,
            )
            network_ms = (time.perf_counter() - t_start) * 1000

            if resp.status_code == 200:
                data = resp.json()
                server_ms = data["inference_time_ms"]
                logger.info(
                    f"Got {len(data['actions'])} actions "
                    f"(server: {server_ms:.0f}ms, total: {network_ms:.0f}ms)"
                )
                return data["actions"]
            else:
                logger.error(f"Server error {resp.status_code}: {resp.text}")
                return None

        except requests.exceptions.Timeout:
            logger.error("Server request timed out")
            return None
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None

    def inference_thread(self, robot):
        """Thread that requests new action chunks from the server."""
        try:
            logger.info("[INFERENCE] Starting inference thread")
            while not self.shutdown_event.is_set():
                # Request new actions when queue is low
                with self.queue_lock:
                    queue_size = len(self.action_queue)

                if queue_size <= 5:
                    # Get observation from robot
                    with self.robot_lock:
                        obs = robot.get_observation()

                    # Share frames with recorder (no extra camera reads needed)
                    if self.record_video:
                        with self.frame_lock:
                            self.latest_frames["front"] = obs.get("front")
                            self.latest_frames["wrist"] = obs.get("wrist")

                    actions = self.request_actions(obs)

                    if actions:
                        with self.queue_lock:
                            self.action_queue.extend(actions)
                            logger.info(f"[INFERENCE] Queue size: {len(self.action_queue)}")
                else:
                    time.sleep(0.05)

            logger.info("[INFERENCE] Thread shutting down")
        except Exception as e:
            logger.error(f"[INFERENCE] Error: {e}")
            import traceback
            traceback.print_exc()
            self.shutdown_event.set()

    def actor_thread(self, robot):
        """Thread that executes actions on the robot."""
        try:
            logger.info("[ACTOR] Starting actor thread")
            action_count = 0

            # Log actions to CSV for analysis
            import csv
            log_file = open("client_action_log.csv", "w", newline="")
            csv_writer = csv.writer(log_file)
            csv_writer.writerow(["step", "time"] + self.joint_names)
            t0 = time.perf_counter()

            while not self.shutdown_event.is_set():
                t_start = time.perf_counter()

                action = None
                with self.queue_lock:
                    if self.action_queue:
                        action = self.action_queue.popleft()

                if action is not None:
                    # Convert to action dict
                    action_dict = {
                        f"{name}.pos": action[i]
                        for i, name in enumerate(self.joint_names)
                    }
                    with self.robot_lock:
                        robot.send_action(action_dict)
                        # Update shared frames for recorder on every action step
                        if self.record_video:
                            obs = robot.get_observation()
                            with self.frame_lock:
                                self.latest_frames["front"] = obs.get("front")
                                self.latest_frames["wrist"] = obs.get("wrist")

                    # Log action
                    csv_writer.writerow(
                        [action_count, f"{time.perf_counter()-t0:.3f}"] + [f"{v:.4f}" for v in action]
                    )
                    log_file.flush()
                    action_count += 1

                    if action_count % 10 == 0:
                        with self.queue_lock:
                            qs = len(self.action_queue)
                        logger.info(f"[ACTOR] Actions executed: {action_count}, queue: {qs}")

                dt = time.perf_counter() - t_start
                sleep_time = self.action_interval - dt
                if sleep_time > 0:
                    time.sleep(sleep_time)

            logger.info(f"[ACTOR] Thread shutting down. Total actions: {action_count}")
        except Exception as e:
            logger.error(f"[ACTOR] Error: {e}")
            import traceback
            traceback.print_exc()
            self.shutdown_event.set()

    def recorder_thread(self, output_path: str):
        """Thread that records side-by-side camera video with modern overlays until shutdown.
        Uses pyav (h264) for encoding — same as lerobot's video pipeline."""
        import av
        from PIL import Image, ImageDraw, ImageFont

        try:
            # Force .mp4 extension for h264
            if not output_path.endswith(".mp4"):
                output_path = output_path.rsplit(".", 1)[0] + ".mp4"
            logger.info(f"[RECORDER] Starting recording to {output_path} (until stopped)")

            # Video settings
            frame_w, frame_h = 640, 480
            gap = 16
            bar_h = 70
            canvas_w, canvas_h = frame_w * 2 + gap, frame_h + bar_h
            out_fps = 15

            # Setup pyav encoder
            container = av.open(output_path, mode="w")
            stream = container.add_stream("libx264", rate=out_fps)
            stream.width = canvas_w
            stream.height = canvas_h
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "18", "preset": "medium"}

            # Load modern fonts
            try:
                font_bold = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf", 32)
                font_medium = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSans-Medium.ttf", 18)
                font_speed = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf", 22)
                font_timer = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSans-Medium.ttf", 22)
            except Exception:
                font_bold = ImageFont.load_default()
                font_medium = ImageFont.load_default()
                font_speed = ImageFont.load_default()
                font_timer = ImageFont.load_default()

            t0 = time.time()
            frame_count = 0

            while not self.shutdown_event.is_set():
                elapsed = time.time() - t0

                # Read shared frames from inference thread (no robot lock needed)
                with self.frame_lock:
                    front_img = self.latest_frames["front"]
                    wrist_img = self.latest_frames["wrist"]

                if front_img is None or wrist_img is None:
                    time.sleep(0.05)
                    continue

                # Camera returns RGB; resize
                front = cv2.resize(front_img, (frame_w, frame_h))
                wrist = cv2.resize(wrist_img, (frame_w, frame_h))

                # Build canvas in RGB (for PIL)
                canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
                canvas[:] = (20, 20, 25)  # dark background

                # Place cameras side by side
                canvas[0:frame_h, 0:frame_w] = front
                canvas[0:frame_h, frame_w + gap:frame_w * 2 + gap] = wrist

                # Switch to PIL for modern text
                pil_img = Image.fromarray(canvas)
                draw = ImageDraw.Draw(pil_img)

                # Semi-transparent overlay behind "SmolVLA" label
                overlay_box = [(8, 6), (175, 44)]
                draw.rounded_rectangle(overlay_box, radius=8, fill=(0, 0, 0, 180))

                # "SmolVLA" — top left, white bold
                draw.text((14, 8), "SmolVLA", font=font_bold, fill=(255, 255, 255))

                # Camera labels — bottom bar
                draw.text((12, frame_h + 10), "Front Camera", font=font_medium, fill=(160, 160, 170))
                draw.text((frame_w + gap + 12, frame_h + 10), "Wrist Camera", font=font_medium, fill=(160, 160, 170))

                # Timer — bottom center
                mins = int(elapsed) // 60
                secs = int(elapsed) % 60
                timer_text = f"{mins:02d}:{secs:02d}"
                t_bbox = draw.textbbox((0, 0), timer_text, font=font_timer)
                t_w = t_bbox[2] - t_bbox[0]
                draw.text((canvas_w // 2 - t_w // 2, frame_h + 38), timer_text,
                          font=font_timer, fill=(200, 200, 210))

                # Playback speed — bottom right: play triangle + "1x"
                speed_x = canvas_w - 80
                speed_y = frame_h + 40
                tri = [(speed_x, speed_y), (speed_x, speed_y + 18), (speed_x + 14, speed_y + 9)]
                draw.polygon(tri, fill=(0, 210, 255))
                draw.text((speed_x + 20, speed_y - 3), "1x", font=font_speed, fill=(0, 210, 255))

                # Encode frame with pyav
                av_frame = av.VideoFrame.from_image(pil_img)
                for packet in stream.encode(av_frame):
                    container.mux(packet)
                frame_count += 1

                # Target ~15 FPS recording
                frame_target_time = frame_count / out_fps
                sleep_t = t0 + frame_target_time - time.time()
                if sleep_t > 0:
                    time.sleep(sleep_t)

            # Flush encoder
            for packet in stream.encode():
                container.mux(packet)
            container.close()
            logger.info(f"[RECORDER] Saved {frame_count} frames ({elapsed:.1f}s) to {output_path}")

        except Exception as e:
            logger.error(f"[RECORDER] Error: {e}")
            import traceback
            traceback.print_exc()

    def run(self):
        """Main entry point."""
        # Check server
        if not self.check_server():
            logger.error("Server not available. Start smolvla_server.py on RunPod first.")
            return

        # Connect robot
        logger.info("Connecting to SO-101...")
        robot = SO101Follower(self.robot_config)
        robot.connect()
        logger.info("Robot connected")

        # Warmup cameras (Logitech needs a few frames to initialize)
        logger.info("Warming up cameras...")
        for _ in range(5):
            robot.get_observation()
            time.sleep(0.3)
        logger.info("Cameras ready")

        try:
            # Start threads
            inf_thread = Thread(
                target=self.inference_thread,
                args=(robot,),
                daemon=True,
                name="Inference",
            )
            act_thread = Thread(
                target=self.actor_thread,
                args=(robot,),
                daemon=True,
                name="Actor",
            )

            # Start recorder if enabled
            rec_thread = None
            if self.record_video:
                rec_thread = Thread(
                    target=self.recorder_thread,
                    args=(self.video_output,),
                    daemon=True,
                    name="Recorder",
                )
                rec_thread.start()

            inf_thread.start()
            act_thread.start()
            logger.info(f"Running for {self.duration}s. Press Ctrl+C to stop.")

            # Handle SIGTERM gracefully (from pkill)
            def handle_signal(signum, frame):
                logger.info(f"Signal {signum} received, shutting down gracefully...")
                self.shutdown_event.set()

            signal.signal(signal.SIGTERM, handle_signal)
            signal.signal(signal.SIGINT, handle_signal)

            start_time = time.time()
            while not self.shutdown_event.is_set():
                elapsed = time.time() - start_time
                if elapsed >= self.duration:
                    logger.info("Duration reached")
                    break
                time.sleep(0.5)

        except KeyboardInterrupt:
            logger.info("Ctrl+C received")
        finally:
            self.shutdown_event.set()
            logger.info("Waiting for recorder to flush...")
            if rec_thread:
                rec_thread.join(timeout=5)
            time.sleep(0.5)
            robot.disconnect()
            logger.info("Robot disconnected. Done.")


import signal

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmolVLA Remote Client for SO-101")
    parser.add_argument("--server_url", type=str, required=True,
                        help="URL of the SmolVLA server (e.g. http://RUNPOD_IP:8000)")
    parser.add_argument("--robot_port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--robot_id", type=str, default="my_awesome_follower_arm")
    parser.add_argument("--front_camera", type=str, default="/dev/video4")
    parser.add_argument("--wrist_camera", type=str, default="/dev/video2")
    parser.add_argument("--task", type=str,
                        default="Grab the brain")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--duration", type=float, default=500.0)
    parser.add_argument("--record", action="store_true", help="Record video from robot cameras")
    parser.add_argument("--video_output", type=str, default="smolvla_eval.mp4")
    parser.add_argument("--record_duration", type=float, default=90.0, help="Video recording duration in seconds")
    args = parser.parse_args()

    client = SmolVLAClient(
        server_url=args.server_url,
        robot_port=args.robot_port,
        robot_id=args.robot_id,
        front_camera=args.front_camera,
        wrist_camera=args.wrist_camera,
        task=args.task,
        fps=args.fps,
        duration=args.duration,
        record_video=args.record,
        video_output=args.video_output,
        record_duration=args.record_duration,
    )
    client.run()
