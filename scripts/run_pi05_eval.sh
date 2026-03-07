#!/bin/bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras="{wrist: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}}" \
  --display_data=true \
  --dataset.repo_id=AdithyaRajendran/eval_pi05_so101_grab_brain_t2 \
  --dataset.single_task="Grab the grey brain toy and place it inside the green container" \
  --dataset.episode_time_s=50 \
  --dataset.num_episodes=10 \
  --policy.path=AdithyaRajendran/pi05_so101_grab_brain_t2 \
  --policy.empty_cameras=1 \
  --dataset.rename_map='{"observation.images.front": "observation.images.base_0_rgb", "observation.images.wrist": "observation.images.left_wrist_0_rgb"}'
