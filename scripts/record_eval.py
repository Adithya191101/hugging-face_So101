"""
Standalone camera recorder for SmolVLA eval.
Records front + wrist cameras side by side with overlays.
Run this alongside smolvla_client.py — completely independent, no interference.

Usage:
    python record_eval.py --front /dev/video0 --wrist /dev/video4 --output eval.mp4
    # Then Ctrl+C to stop
"""

import argparse
import time

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def record(front_dev: str, wrist_dev: str, output: str, fps: int = 15):
    # Open cameras directly
    cap_front = cv2.VideoCapture(front_dev)
    cap_wrist = cv2.VideoCapture(wrist_dev)

    cap_front.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_front.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap_wrist.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_wrist.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Warmup
    for _ in range(10):
        cap_front.read()
        cap_wrist.read()
    print("Cameras ready")

    frame_w, frame_h = 640, 480
    gap = 16
    bar_h = 70
    canvas_w = frame_w * 2 + gap
    canvas_h = frame_h + bar_h

    # Setup pyav encoder
    if not output.endswith(".mp4"):
        output = output.rsplit(".", 1)[0] + ".mp4"

    container = av.open(output, mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = canvas_w
    stream.height = canvas_h
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "medium"}

    # Load fonts
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
    print(f"Recording to {output} at {fps} FPS. Press Ctrl+C to stop.")

    try:
        while True:
            elapsed = time.time() - t0

            ret1, front_bgr = cap_front.read()
            ret2, wrist_bgr = cap_wrist.read()
            if not ret1 or not ret2:
                time.sleep(0.01)
                continue

            # Convert BGR to RGB
            front = cv2.cvtColor(cv2.resize(front_bgr, (frame_w, frame_h)), cv2.COLOR_BGR2RGB)
            wrist = cv2.cvtColor(cv2.resize(wrist_bgr, (frame_w, frame_h)), cv2.COLOR_BGR2RGB)

            # Build canvas
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            canvas[:] = (20, 20, 25)
            canvas[0:frame_h, 0:frame_w] = front
            canvas[0:frame_h, frame_w + gap:frame_w * 2 + gap] = wrist

            # PIL overlays
            pil_img = Image.fromarray(canvas)
            draw = ImageDraw.Draw(pil_img)

            # SmolVLA label with background
            draw.rounded_rectangle([(8, 6), (175, 44)], radius=8, fill=(0, 0, 0, 180))
            draw.text((14, 8), "SmolVLA", font=font_bold, fill=(255, 255, 255))

            # Camera labels
            draw.text((12, frame_h + 10), "Front Camera", font=font_medium, fill=(160, 160, 170))
            draw.text((frame_w + gap + 12, frame_h + 10), "Wrist Camera", font=font_medium, fill=(160, 160, 170))

            # Timer
            mins = int(elapsed) // 60
            secs = int(elapsed) % 60
            timer_text = f"{mins:02d}:{secs:02d}"
            t_bbox = draw.textbbox((0, 0), timer_text, font=font_timer)
            t_w = t_bbox[2] - t_bbox[0]
            draw.text((canvas_w // 2 - t_w // 2, frame_h + 38), timer_text,
                      font=font_timer, fill=(200, 200, 210))

            # Play speed
            speed_x = canvas_w - 80
            speed_y = frame_h + 40
            tri = [(speed_x, speed_y), (speed_x, speed_y + 18), (speed_x + 14, speed_y + 9)]
            draw.polygon(tri, fill=(0, 210, 255))
            draw.text((speed_x + 20, speed_y - 3), "1x", font=font_speed, fill=(0, 210, 255))

            # Encode
            av_frame = av.VideoFrame.from_image(pil_img)
            for packet in stream.encode(av_frame):
                container.mux(packet)
            frame_count += 1

            # FPS control
            target_time = frame_count / fps
            sleep_t = t0 + target_time - time.time()
            if sleep_t > 0:
                time.sleep(sleep_t)

            if frame_count % (fps * 5) == 0:
                print(f"  Recording... {mins:02d}:{secs:02d} ({frame_count} frames)")

    except KeyboardInterrupt:
        print("\nStopping...")

    # Flush and close
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    cap_front.release()
    cap_wrist.release()

    elapsed = time.time() - t0
    print(f"Saved {frame_count} frames ({elapsed:.1f}s) to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record eval video from robot cameras")
    parser.add_argument("--front", type=str, default="/dev/video0")
    parser.add_argument("--wrist", type=str, default="/dev/video4")
    parser.add_argument("--output", type=str, default="smolvla_eval.mp4")
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    record(args.front, args.wrist, args.output, args.fps)
