# Setup Guide - Command Line Reference

This document contains all the command-line steps and structure used in this project.

## Project Structure

```
robot-vla-project/
├── README.md                          # Main project documentation
├── docs/                              # Detailed documentation
│   ├── challenges_and_solutions.md   # All problems faced with solutions
│   ├── technical_details.md          # Complete technical reference
│   └── SETUP_GUIDE.md                # This file - command reference
├── videos/                            # Demo videos (to be added)
├── results/                           # Training curves and metrics
├── configs/                           # Configuration files
└── scripts/                           # Training and deployment scripts
```

---

## Environment Setup

### 1. Create Conda Environment

```bash
conda create -n lerobot python=3.10
conda activate lerobot
```

### 2. Install LeRobot

```bash
cd ~
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .
```

### 3. Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install transformers datasets huggingface_hub
pip install wandb opencv-python
```

---

## ACT Model Training

### Recording Demonstration Data

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras="{camera1: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}, camera2: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}}" \
  --display_data=true \
  --dataset.single_task="Pick and place soft object" \
  --dataset.repo_id=AdithyaRajendran/so101_pick_place_act \
  --dataset.episode_time_s=50 \
  --dataset.num_episodes=241 \
  --dataset.fps=30 \
  --dataset.push_to_hub=true \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=my_awesome_leader_arm \
  --dataset.video=true
```

**What I did:**
- Recorded 241 successful episodes
- Demonstrated pick-and-place of soft irregular objects
- Varied object positions for generalization

### Training ACT Model

```bash
cd lerobot
python src/lerobot/scripts/lerobot_train.py \
  --policy.path=lerobot/act \
  --dataset.repo_id=AdithyaRajendran/so101_pick_place_act \
  --batch_size=8 \
  --num_workers=4 \
  --steps=100000 \
  --policy.chunk_size=100 \
  --policy.n_action_steps=100 \
  --optimizer.lr=1e-5 \
  --output_dir=./checkpoints/act_policy \
  --job_name=so101_act_policy \
  --policy.repo_id=AdithyaRajendran/so101_act_policy \
  --wandb.enable=true \
  --wandb.project=lerobot
```

**Results achieved:**
- **80% success rate** on pick-and-place tasks
- Soft irregular round/obloid objects
- 241 training episodes, 100K+ frames

---

## SmolVLA Model Training

### Recording Demonstration Data

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras="{camera1: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}, camera2: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}}" \
  --display_data=true \
  --dataset.single_task="Grab the brain" \
  --dataset.repo_id=AdithyaRajendran/so101_grab_brain_t2 \
  --dataset.episode_time_s=50 \
  --dataset.num_episodes=241 \
  --dataset.fps=30 \
  --dataset.push_to_hub=true \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=my_awesome_leader_arm \
  --dataset.video=true
```

**What I did:**
- Recorded 241 successful grasping episodes
- Task: "Grab the brain" (language-conditioned)
- 2×2 inch pickup area

### Training SmolVLA v5 (Optimized)

```bash
cd /content/lerobot
python src/lerobot/scripts/lerobot_train.py \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=AdithyaRajendran/so101_grab_brain_t2 \
  --dataset.image_transforms.enable=true \
  --batch_size=32 \
  --num_workers=2 \
  --steps=20000 \
  --policy.use_amp=true \
  --optimizer.lr=1e-05 \
  --optimizer.grad_clip_norm=10.0 \
  --policy.chunk_size=10 \
  --policy.n_action_steps=10 \
  --policy.device=cuda \
  --output_dir=/content/drive/MyDrive/lerobot_checkpoints/so101_smolvla_v5 \
  --job_name=so101_smolvla_policy_FINAL_v5 \
  --policy.repo_id=AdithyaRajendran/so101_smolvla_policy_FINAL_v5 \
  --policy.empty_cameras=1 \
  --rename_map='{"observation.images.front":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}' \
  --save_freq=2500 \
  --eval_freq=1250 \
  --wandb.enable=true \
  --wandb.project=lerobot \
  --wandb.name=so101_smolvla_FINAL_v5
```

**Key parameters:**
- steps=20,000 → 6.35 epochs (optimal, prevents overfitting)
- batch_size=32 (GPU efficient)
- chunk_size=10, n_action_steps=10 (reactive but jerky)
- Training time: ~3.5 hours on V100 GPU

### Training SmolVLA v6 (Planned - Smooth + Reactive)

```bash
cd /content/lerobot
python src/lerobot/scripts/lerobot_train.py \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=AdithyaRajendran/so101_grab_brain_merged \
  --dataset.image_transforms.enable=true \
  --batch_size=32 \
  --num_workers=2 \
  --steps=25000 \
  --policy.use_amp=true \
  --optimizer.lr=1e-05 \
  --optimizer.grad_clip_norm=10.0 \
  --policy.chunk_size=30 \
  --policy.n_action_steps=20 \
  --policy.device=cuda \
  --output_dir=/content/drive/MyDrive/lerobot_checkpoints/so101_smolvla_v6 \
  --job_name=so101_smolvla_policy_FINAL_v6 \
  --policy.repo_id=AdithyaRajendran/so101_smolvla_policy_FINAL_v6 \
  --policy.empty_cameras=1 \
  --rename_map='{"observation.images.front":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}' \
  --save_freq=2500 \
  --eval_freq=1250 \
  --wandb.enable=true \
  --wandb.project=lerobot \
  --wandb.name=so101_smolvla_FINAL_v6
```

**Improvements in v6:**
- chunk_size=30 → smoother motion (1 second prediction window)
- n_action_steps=20 → better balance smooth/reactive
- Mixed dataset (291 episodes) for better generalization

---

## Model Deployment & Testing

### Deploy SmolVLA Policy

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras="{camera1: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}, camera2: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}}" \
  --display_data=true \
  --dataset.single_task="Grab the brain" \
  --dataset.repo_id=AdithyaRajendran/eval_so101_smolvla_policy_FINAL_v5 \
  --dataset.episode_time_s=50 \
  --dataset.num_episodes=10 \
  --policy.path=AdithyaRajendran/so101_smolvla_policy_FINAL_v5 \
  --policy.empty_cameras=1 \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=my_awesome_leader_arm
```

**CRITICAL: Language instruction must match exactly!**
- Training: `"Grab the brain"`
- Deployment: `"Grab the brain"` (MUST be identical)
- Even "Grasp a brain and put it in the bin" fails completely!

### Verify Camera Configuration

```bash
# Always verify cameras visually before deployment
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras="{camera1: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}, camera2: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}}" \
  --display_data=true \
  --dataset.single_task="test" \
  --dataset.repo_id=test_camera_verify \
  --dataset.episode_time_s=10 \
  --dataset.num_episodes=1 \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=my_awesome_leader_arm
```

**Check:**
- camera1 window should show FRONT view (table/scene)
- camera2 window should show WRIST view (gripper camera)
- If swapped, swap /dev/video4 and /dev/video2

---

## Debugging & Analysis

### Analyze Dataset Statistics

```python
from datasets import load_dataset
import numpy as np

# Load dataset
ds = load_dataset("AdithyaRajendran/so101_grab_brain_t2", split="train")

# Analyze gripper actions
gripper_actions = ds["action"][:, 5]  # Gripper is index 5
print(f"Gripper mean: {np.mean(gripper_actions):.2f}")
print(f"Gripper closed (<0.2): {np.mean(gripper_actions < 0.2) * 100:.1f}%")
print(f"Gripper open (>0.8): {np.mean(gripper_actions > 0.8) * 100:.1f}%")

# Calculate epochs
total_frames = len(ds)
batch_size = 32
steps = 20000
steps_per_epoch = total_frames / batch_size
epochs = steps / steps_per_epoch
print(f"\nTraining epochs: {epochs:.2f}")
```

### Compare Training vs Deployment

```python
# Training dataset
train_ds = load_dataset("AdithyaRajendran/so101_grab_brain_t2", split="train")
train_gripper = train_ds["action"][:, 5]

# Deployment dataset
eval_ds = load_dataset("AdithyaRajendran/eval_so101_smolvla_policy_FINAL_v5", split="train")
eval_gripper = eval_ds["action"][:, 5]

print(f"Training gripper mean: {np.mean(train_gripper):.2f}")
print(f"Deployment gripper mean: {np.mean(eval_gripper):.2f}")
print(f"\nTraining closed %: {np.mean(train_gripper < 0.2) * 100:.1f}%")
print(f"Deployment closed %: {np.mean(eval_gripper < 0.2) * 100:.1f}%")
```

---

## Problems I Faced & Solutions

### Problem 1: Language Instruction Mismatch ⚠️ CRITICAL

**What happened:**
- Robot went near object but gripper NEVER closed (0% success)
- Training showed 54.5% gripper closed, deployment showed 0%

**Root cause:**
- Training used: `"Grab the brain"`
- Deployment used: `"Grasp a brain and put it in the bin."`
- VLA models create different embeddings for different text!

**Solution:**
```bash
# Must use EXACT same task description
--dataset.single_task="Grab the brain"  # Match training exactly!
```

**Result:** 0% → 33% success rate improvement

---

### Problem 2: Camera Configuration Swap ⚠️ CRITICAL

**What happened:**
- After environment adjustment, robot crashed into BIN instead of object
- All models (v3, v4, v5) failed identically

**Root cause:**
- Physical cameras swapped: camera1 showing wrist, camera2 showing front
- Model's spatial reasoning completely inverted

**Solution:**
```bash
# Verify cameras visually first
--display_data=true

# If swapped, swap device assignments
--robot.cameras="{camera1: {index_or_path: /dev/video2}, camera2: {index_or_path: /dev/video4}}"
```

**Result:** Robot approached object correctly again

---

### Problem 3: Overfitting on Small Dataset

**What happened:**
- Training for 50,000 steps gave very low loss (0.014)
- Poor generalization in deployment

**Root cause:**
- 50,000 steps = 15.87 epochs on 241 episodes
- Industry standard: 3-7 epochs for <500 episodes

**Solution:**
```bash
--steps=20000  # Reduced from 50,000
# Result: 6.35 epochs, loss 0.012, better generalization
```

**Result:** 2.5x faster training (9h → 3.5h) with better performance

---

### Problem 4: Action Smoothness (Jerky Motion)

**What happened:**
- v5 showed jerky motion with 10-26° discontinuities
- Training data only had 3-13° jumps

**Root cause:**
- chunk_size=10, n_action_steps=10
- Re-plans every 0.33 seconds → discrete jumps

**Solution:**
```bash
# v6 configuration
--policy.chunk_size=30       # Predict 1 second ahead (smooth)
--policy.n_action_steps=20   # Execute 0.67s before re-plan (reactive)
```

**Result:** Smoother motion with adequate reactivity

---

### Problem 5: Starting State Inconsistency

**What happened:**
- Performance varied 0-33% between episodes
- Robot started from random positions

**Root cause:**
- Follower arm doesn't auto-reset
- Training started with gripper CLOSED (0.5°)
- Deployment started with gripper OPEN (2.8-40°)

**Solution:**
```bash
# Manual reset protocol before EACH episode:
# Use leader arm to position follower to:
#   shoulder_pan: -8°
#   shoulder_lift: -98°
#   elbow_flex: 100°
#   gripper: 0.5° (CLOSED!)
```

**Result:** Improved consistency across episodes

---

### Problem 6: Visual Distribution Shift (Ongoing)

**What happened:**
- Even with all fixes, still poor performance
- Training environment ≠ deployment environment

**Root cause:**
- Different camera viewpoints
- Different object placement distribution
- Model overfitted to training visuals

**Solution in progress:**
```bash
# Mixed-dataset approach:
# 1. Record 50 episodes in deployment environment
lerobot-record --dataset.repo_id=deployment_mix --num_episodes=50

# 2. Train v6 on merged dataset
python lerobot_train.py \
  --dataset.repo_id=AdithyaRajendran/so101_grab_brain_merged \
  --steps=25000
```

**Expected result:** >70% success rate, 6×6 inch pickup area

---

## Key Learnings

1. **Configuration consistency is CRITICAL**
   - Language instructions must match exactly
   - Camera assignments must be identical
   - Starting states must be consistent

2. **Quantitative debugging is essential**
   - Analyze action distributions (54.5% → 0% revealed language issue)
   - Calculate actual epochs, not just steps
   - Compare training vs deployment statistics

3. **Understand model internals**
   - VLAs create different embeddings for different text
   - Visual pipeline must be reproducible
   - Camera swap inverts spatial reasoning

4. **Data-centric approach**
   - Training data quality > model complexity
   - Multi-environment datasets improve generalization
   - Document environment setup with checkpoints

---

## HuggingFace Links

### Datasets
- Training: https://huggingface.co/datasets/AdithyaRajendran/so101_grab_brain_t2
- Evaluation v4: https://huggingface.co/datasets/AdithyaRajendran/eval_so101_smolvla_policy_FINAL_v4
- Evaluation v5: https://huggingface.co/datasets/AdithyaRajendran/eval_so101_smolvla_policy_FINAL_v5

### Models
- SmolVLA v5: https://huggingface.co/AdithyaRajendran/so101_smolvla_policy_FINAL_v5
- SmolVLA v6: https://huggingface.co/AdithyaRajendran/so101_smolvla_policy_FINAL_v6 (in progress)

### Weights & Biases
- Project: https://wandb.ai/[your-username]/lerobot

---

## What I Accomplished

✅ Implemented ACT model with **80% success rate** on soft irregular objects
✅ Trained SmolVLA (500M parameters) for language-conditioned manipulation
✅ Debugged 6+ critical deployment failures systematically
✅ Optimized training (2.5x faster) while improving generalization
✅ Developed quantitative debugging methodology
✅ Published datasets and models to HuggingFace Hub
✅ Documented all problems and solutions for reproducibility
🔄 Working on mixed-dataset training for visual generalization (v6)

---

## Next Steps

1. Record 50 deployment mix episodes
2. Train v6 on merged dataset (291 episodes)
3. Achieve >70% success rate on 6×6 inch pickup area
4. Add demo videos to repository
5. Export W&B training curves as images
