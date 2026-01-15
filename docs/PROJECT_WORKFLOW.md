# Project Workflow - Step by Step

This document outlines the complete workflow I followed in this project, from data collection to deployment.

## Timeline: December 2024 - January 2025

---

## Phase 1: ACT Model Implementation (Weeks 1-2)

### Step 1: Environment Setup
```bash
# Install LeRobot framework
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .
```

### Step 2: Data Collection
- Recorded 241 demonstration episodes using teleoperation
- Task: Pick and place soft irregular round/obloid objects
- Used SO-101 leader arm for teleoperation
- Collected dual-camera data (front + wrist views)

**Command used:**
```bash
lerobot-record \
  --robot.type=so101_follower \
  --dataset.single_task="Pick and place soft object" \
  --dataset.num_episodes=241 \
  --dataset.push_to_hub=true
```

### Step 3: ACT Training
- Trained Action Chunking Transformer model
- 100,000 training steps
- chunk_size=100 for smooth trajectories

**Result:** ✅ **80% success rate** on pick-and-place tasks

---

## Phase 2: SmolVLA Model Implementation (Weeks 3-4)

### Step 1: Data Collection for Language-Conditioned Task
- Recorded 241 new episodes for grasping task
- Task description: **"Grab the brain"**
- Focused on 2×2 inch pickup area
- Total: 100,832 frames at 30 FPS

**Dataset:** `AdithyaRajendran/so101_grab_brain_t2`

### Step 2: Initial Training (v2, v3, v4)
- Trained multiple versions experimenting with hyperparameters
- Used SmolVLM2-500M-Video-Instruct base model
- chunk_size=30, n_action_steps=30

**Training command:**
```bash
python lerobot_train.py \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=AdithyaRajendran/so101_grab_brain_t2 \
  --steps=50000 \
  --batch_size=32
```

**Problem encountered:** Models showed low training loss but failed in deployment

---

## Phase 3: Debugging & Problem Solving (Weeks 5-6)

### Problem 1: 0% Deployment Success

**Symptoms:**
- Robot approached object but gripper NEVER closed
- Training: 54.5% gripper closed
- Deployment: 0% gripper closed

**Steps I took:**

1. **Analyzed action distributions:**
```python
# Discovered gripper mean: 6.83 (training) vs 25-26 (deployment)
from datasets import load_dataset
ds = load_dataset("AdithyaRajendran/so101_grab_brain_t2")
gripper_actions = ds["action"][:, 5]
print(f"Gripper closed: {np.mean(gripper_actions < 0.2) * 100}%")
# Output: 54.5%

eval_ds = load_dataset("AdithyaRajendran/eval_so101_smolvla_policy_FINAL_v4")
eval_gripper = eval_ds["action"][:, 5]
print(f"Gripper closed: {np.mean(eval_gripper < 0.2) * 100}%")
# Output: 0.0%  ⚠️ PROBLEM IDENTIFIED!
```

2. **Checked dataset metadata:**
```python
# Found training task: "Grab the brain"
# But deployment used: "Grasp a brain and put it in the bin."
# LANGUAGE MISMATCH! ⚠️
```

3. **Fixed language instruction:**
```bash
--dataset.single_task="Grab the brain"  # Now matches training exactly
```

**Result:** 0% → 33% success rate improvement

---

### Problem 2: Robot Crashes Into Bin

**What happened:**
- Tried to adjust environment to match training
- Robot behavior got WORSE - crashed into bin instead of object
- All models (v3, v4, v5) failed the same way

**Steps I took:**

1. **Suspected camera swap:**
   - Hypothesis: camera1 and camera2 physically swapped

2. **Verified camera configuration:**
```bash
lerobot-record --display_data=true
# Checked camera1 window: showed WRIST view (should be FRONT)
# Checked camera2 window: showed FRONT view (should be WRIST)
# CAMERAS WERE SWAPPED! ⚠️
```

3. **Fixed camera mapping:**
```bash
# Swapped /dev/video4 and /dev/video2
--robot.cameras="{camera1: {index_or_path: /dev/video2}, camera2: {index_or_path: /dev/video4}}"
```

**Result:** Robot approached object correctly again

---

### Problem 3: Overfitting

**What I noticed:**
- Very low training loss (0.014)
- But poor generalization

**Steps I took:**

1. **Calculated actual epochs:**
```python
total_frames = 100832
batch_size = 32
steps = 50000

steps_per_epoch = total_frames / batch_size  # 3,151
epochs = steps / steps_per_epoch
print(f"Epochs: {epochs}")
# Output: 15.87 epochs ⚠️ WAY TOO MANY!
```

2. **Researched industry standards:**
   - <500 episodes: 3-7 epochs typical
   - 15.87 epochs = severe overfitting

3. **Optimized training duration:**
```bash
# v5 retraining
--steps=20000  # Down from 50,000
# Result: 6.35 epochs ✅ OPTIMAL
```

**Result:** 2.5x faster training (9h → 3.5h), better generalization

---

### Problem 4: Jerky Motion

**What I observed:**
- v5 model showed discontinuous motion
- 10-26° jumps between frames
- Training data only had 3-13° jumps

**Steps I took:**

1. **Analyzed action smoothness:**
```python
# Calculated action differences frame-to-frame
action_diffs = np.diff(actions, axis=0)
max_jumps = np.max(np.abs(action_diffs), axis=0)
print(f"Max jumps: {max_jumps}")
# shoulder_lift: 26° (deployment) vs 13° (training)
```

2. **Identified root cause:**
   - chunk_size=10 → re-plans every 0.33s
   - Too frequent re-planning → jerky motion

3. **Designed v6 configuration:**
```bash
--policy.chunk_size=30       # Predict 1 second ahead
--policy.n_action_steps=20   # Execute 0.67s before re-plan
```

**Expected result:** Smoother motion with adequate reactivity

---

### Problem 5: Starting State Inconsistency

**What I noticed:**
- Success rate varied 0-33% between episodes
- Robot started from different positions each time

**Steps I took:**

1. **Analyzed starting states:**
```python
# Training starting positions:
# shoulder_pan: -8° ± 3°
# gripper: 0.5° ± 0.3° (CLOSED)

# Deployment starting positions:
# shoulder_pan: -145° to 14° (huge variation!)
# gripper: 2.8° to 40° (OPEN!)
```

2. **Implemented manual reset protocol:**
   - Use leader arm before each episode
   - Match training starting state exactly
   - Especially gripper CLOSED (0.5°)

**Result:** Improved consistency

---

## Phase 4: SmolVLA v5 Training (Week 6)

### Optimized Training Run

**Parameters:**
```bash
python lerobot_train.py \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=AdithyaRajendran/so101_grab_brain_t2 \
  --steps=20000 \
  --batch_size=32 \
  --policy.chunk_size=10 \
  --policy.n_action_steps=10 \
  --optimizer.lr=1e-05 \
  --rename_map='{"observation.images.front":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'
```

**Training stats:**
- Time: 3.5 hours on V100 GPU
- Final loss: 0.012
- Epochs: 6.35 ✅
- GPU utilization: 90-95%

**Deployment results:**
- Success rate: 33% (after language fix)
- Gripper closed: 20-30% of frames (improved from 0%)
- Still room for improvement

---

## Phase 5: Current Work - SmolVLA v6 (In Progress)

### Mixed-Dataset Approach

**Plan:**
1. Record 50 new episodes in deployment environment
2. Merge with 241 original episodes → 291 total
3. Train v6 on combined dataset

**Why this approach:**
- Learns to work in BOTH environments
- Improves visual generalization
- Doesn't waste existing data

**v6 Configuration:**
```bash
python lerobot_train.py \
  --dataset.repo_id=AdithyaRajendran/so101_grab_brain_merged \
  --steps=25000 \
  --policy.chunk_size=30 \
  --policy.n_action_steps=20
```

**Expected improvements:**
- Smoother motion (chunk_size=30)
- Better generalization (mixed dataset)
- >70% success rate target
- 6×6 inch pickup area (from 2×2 inch)

---

## Summary of What I Did

### Data Collection
✅ Recorded 241 episodes for ACT (pick-and-place)
✅ Recorded 241 episodes for SmolVLA (language-conditioned grasping)
✅ Total: ~200,000 frames collected via teleoperation

### Model Training
✅ Trained ACT model successfully (80% success rate)
✅ Trained SmolVLA v2, v3, v4, v5 (iterative improvements)
✅ Used Google Colab with V100/A100 GPUs
✅ Tracked experiments with Weights & Biases

### Debugging & Problem Solving
✅ Identified 6+ critical deployment failures
✅ Developed quantitative debugging methodology
✅ Fixed language instruction mismatch (0% → 33% improvement)
✅ Fixed camera configuration swap
✅ Optimized training to prevent overfitting (2.5x speedup)
✅ Balanced action smoothness vs reactivity

### Documentation & Sharing
✅ Published datasets to HuggingFace Hub
✅ Published model checkpoints to HuggingFace Hub
✅ Documented all problems and solutions
✅ Created reproducible setup guides
✅ Tracked experiments with W&B

---

## Tools & Technologies Used

### Frameworks
- **LeRobot** - End-to-end robot learning pipeline
- **PyTorch** - Deep learning framework
- **HuggingFace** - Transformers, Datasets, Hub

### Cloud & Infrastructure
- **Google Colab** - V100/A100 GPU training
- **Weights & Biases** - Experiment tracking
- **HuggingFace Hub** - Dataset and model hosting

### Computer Vision
- **OpenCV** - Camera interface and processing
- **SmolVLM2** - Vision-language model base

### Hardware
- **SO-101 Follower Arm** - 6-DOF robot arm
- **SO-101 Leader Arm** - Teleoperation device
- **Dual cameras** - Front (scene) + wrist (gripper)

---

## Key Metrics

### ACT Model
- **Success rate:** 80%
- **Task:** Pick and place soft irregular objects
- **Training data:** 241 episodes
- **Training time:** ~8 hours

### SmolVLA Model
- **Current success rate:** 33% (v5 with language fix)
- **Target success rate:** >70% (v6 in progress)
- **Task:** "Grab the brain" (language-conditioned)
- **Model size:** 500M parameters
- **Training data:** 241 episodes (100,832 frames)
- **Workspace:** 2×2 inch → targeting 6×6 inch

### Training Optimization
- **Original:** 50,000 steps, 15.87 epochs, 9 hours
- **Optimized:** 20,000 steps, 6.35 epochs, 3.5 hours
- **Speedup:** 2.5x faster training
- **Result:** Better generalization with lower training time

---

## Lessons Learned

1. **Configuration consistency is everything**
   - Language instructions must match exactly
   - Camera assignments must be identical
   - Starting states significantly impact performance

2. **Quantitative debugging beats trial-and-error**
   - Action distribution analysis revealed language mismatch
   - Epoch calculation identified overfitting
   - Statistical comparison guided solutions

3. **Understanding model internals is critical**
   - VLAs create different embeddings for different text
   - Visual observation pipeline must be reproducible
   - Spatial reasoning breaks completely with camera swap

4. **Small datasets need careful handling**
   - Monitor epochs, not just steps
   - 3-7 epochs optimal for <500 episodes
   - Data augmentation helps but doesn't fix all issues

5. **Trade-offs are fundamental**
   - Smoothness vs reactivity (chunk_size vs n_action_steps)
   - Training time vs generalization (epochs)
   - GPU efficiency vs gradient noise (batch_size)

---

## Future Work

1. **Complete v6 training** with mixed-dataset approach
2. **Add demo videos** showing successful deployments
3. **Expand pickup area** from 2×2 inch to 6×6 inch
4. **Test on multiple object types** for generalization
5. **Implement automatic reset** mechanism for consistency
6. **Explore other VLA architectures** (OpenVLA, RT-2)

---

## Contact & Links

**GitHub:** https://github.com/Adithya191101/hugging-face_So101
**HuggingFace:** https://huggingface.co/AdithyaRajendran
**LinkedIn:** [Add your LinkedIn]
**Email:** [Add your email]
