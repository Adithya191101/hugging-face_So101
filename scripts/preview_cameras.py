#!/usr/bin/env python3
"""Live preview of front camera using matplotlib."""

import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

cap = cv2.VideoCapture("/dev/video4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Failed to open /dev/video4")
    exit(1)

print("Front camera live preview. Close the window to quit.")

plt.ion()
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title("FRONT CAMERA - Live")
ax.axis('off')

ret, frame = cap.read()
img_plot = ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

try:
    while plt.fignum_exists(fig.number):
        ret, frame = cap.read()
        if ret:
            img_plot.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.03)
        else:
            break
except KeyboardInterrupt:
    pass

cap.release()
plt.close()
