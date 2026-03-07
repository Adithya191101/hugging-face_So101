#!/bin/bash
# Evaluate SmolVLA v9 on SO-101 robot using RTC
# Run from: cd ~/lerobot && bash run_smolvla_eval.sh

set -e

POLICY_REPO="AdithyaRajendran/smolvla_so101_grab_brain_t2_v9"
TASK="Grab the grey brain toy and place it inside the green container"

echo "=== SmolVLA v9 Evaluation (RTC) ==="
echo "Policy: $POLICY_REPO"
echo "Task: $TASK"
echo ""

python examples/rtc/eval_with_real_robot.py \
  --policy.path="$POLICY_REPO" \
  --policy.device=cuda \
  --policy.num_steps=5 \
  --policy.use_amp=true \
  --rtc.enabled=true \
  --rtc.execution_horizon=10 \
  --rtc.prefix_attention_schedule=EXP \
  --rtc.max_guidance_weight=5.0 \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras='{"wrist": {"type": "opencv", "index_or_path": "/dev/video2", "width": 640, "height": 480, "fps": 30}, "front": {"type": "opencv", "index_or_path": "/dev/video4", "width": 640, "height": 480, "fps": 30}}' \
  --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --task="$TASK" \
  --fps=10 \
  --duration=500 \
  --display_data=true
