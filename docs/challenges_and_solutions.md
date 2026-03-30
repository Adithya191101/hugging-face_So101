# Challenges & Solutions - Robot Learning Project

## Overview

This document details the major technical challenges encountered during the implementation of vision-language-action models for robotic manipulation, along with systematic solutions and key learnings.

---

## Challenge 1: Language Instruction Mismatch (Critical)

### Problem Description
After training SmolVLA model (v2-v5) with 241 episodes, robot would approach the object but **gripper never closed** during deployment (0% success rate).

**Symptoms:**
- Training data: 54.5% of frames had gripper closed
- Deployment: 0% of frames had gripper closed
- Gripper action mean: 6.83 (training) vs 25-26 (deployment)
- Model output stuck in narrow range (+1.5 normalized value) instead of full range (-0.54 to +3.25)

### Investigation Process

1. **Initial Hypothesis**: Overfitting
   - Analyzed training curves: 15.87 epochs on 241 episodes
   - Found: Excessive training, but not the root cause

2. **Second Hypothesis**: Chunk size too large (30 timesteps)
   - Retrained v5 with chunk_size=10 for more reactive control
   - Result: Slightly better but still 0% success rate

3. **Root Cause Discovery**:
   - Checked dataset metadata: Training task was **"Grab the brain"**
   - Checked deployment config: Using **"Grasp a brain and put it in the bin."**
   - **MISMATCH FOUND!**

### Solution

**Vision-language models create different embeddings for different text**. Model learned:
```
"Grab the brain" → [embedding_1] → {approach, close gripper, lift}
```

But during deployment received:
```
"Grasp a brain and put it in the bin." → [embedding_2] ≠ [embedding_1] → {default safe behavior: keep gripper open}
```

**Fix**: Use **exact same task description** in deployment as training:
```bash
--dataset.single_task="Grab the brain"  # Must match training exactly!
```

**Result**: Success rate improved from 0% to 33% (1 out of 3 episodes)

### Key Learnings

1. **Language conditioning is EXACT**: Even semantically similar phrases create different embeddings
2. **Always verify**: Check dataset metadata vs deployment config
3. **VLA models are sensitive**: Small text differences cause complete behavioral changes
4. **Documentation critical**: Must document exact training task for reproducibility

**Impact**: Critical bug that would have been impossible to debug without understanding VLA architecture

---

## Challenge 2: Camera Configuration Mismatch (Critical)

### Problem Description
After attempting to match training environment setup, robot behavior got **dramatically worse** - crashed into bin instead of approaching object. All models (v3, v4, v5) failed identically.

**Symptoms:**
- Original setup: Robot approached object (partial success)
- After camera adjustment: Robot crashed into bin (complete failure)
- All trained models showed same failure pattern
- Model seemed to confuse object location with bin location

### Investigation Process

1. **Initial Hypothesis**: Environmental change broke visual features
   - Checked lighting, table position, camera angles
   - Everything seemed correct

2. **User Insight**: "I think video output been changed, camera1 as wrist and 2 as front"

3. **Verification**:
   - Ran test with `--display_data=true` to visualize camera feeds
   - **Confirmed**: camera1 (expected front view) showed wrist camera
   - camera2 (expected wrist view) showed front camera
   - **Cameras were swapped!**

### Technical Explanation

Training dataset used rename mapping:
```python
rename_map = {
    "observation.images.front": "observation.images.camera1",
    "observation.images.wrist": "observation.images.camera2"
}
```

Model learned spatial reasoning based on:
- camera1 = front view (scene overview, object localization)
- camera2 = wrist view (fine-grained gripper control)

When cameras swapped during deployment:
- Model received wrist view in camera1 → tried to use it for scene understanding
- Model received front view in camera2 → tried to use it for gripper control
- **Spatial understanding completely inverted** → robot targeted wrong objects

### Solution

**Diagnostic test**:
```bash
lerobot-record --display_data=true  # Visually verify camera assignments
```

**Fix**: Swap video device mappings to match training:
```bash
# Original (WRONG):
--robot.cameras="{camera1: {index_or_path: /dev/video4}, camera2: {index_or_path: /dev/video2}}"

# Corrected (RIGHT):
--robot.cameras="{camera1: {index_or_path: /dev/video2}, camera2: {index_or_path: /dev/video4}}"
```

**Result**: Robot approached object correctly again

### Key Learnings

1. **Camera consistency is critical**: Physical camera assignments must match training exactly
2. **Always verify visually**: Use display mode to confirm camera feeds before deployment
3. **Document hardware setup**: Save exact camera configuration with training data
4. **Spatial reasoning fragile**: Even small observation mismatches cause catastrophic failures

**Impact**: Without this fix, no amount of retraining would have worked

---

## Challenge 3: Overfitting on Small Dataset (Moderate)

### Problem Description
Training for 50,000 steps resulted in very low loss (0.014) but poor generalization during deployment.

**Symptoms:**
- Training loss decreased from 0.062 → 0.014 (very low)
- Training: 15.87 epochs on 241 episodes
- Model memorized specific trajectories instead of learning general grasping strategy
- Deployment: Inconsistent behavior across slight object position variations

### Analysis

**Industry standards for robot learning datasets**:
- <500 episodes: 3-7 epochs typical
- 500-1000 episodes: 5-10 epochs typical
- Our training: **15.87 epochs on 241 episodes = 2-3x too many**

**Mathematical calculation**:
```
Steps per epoch = 100,832 frames / 32 batch_size = 3,151 steps/epoch
50,000 steps / 3,151 = 15.87 epochs WARNING:  TOO MANY
```

**What happens with overfitting**:
- Model memorizes exact camera views and trajectories from training
- Fails to generalize to slight variations in object positions
- Produces precise actions for memorized scenes, conservative actions for novel scenes
- Loss continues decreasing without plateau (red flag)

### Solution

**Optimal training duration**:
```
Target: 6-7 epochs for 241-episode dataset
6.35 epochs × 3,151 steps/epoch = 20,000 steps 
```

**v5 Retraining configuration**:
```bash
--steps=20000  # Reduced from 50,000
# Result: 6.35 epochs, final loss ~0.012
```

**Additional regularization**:
```bash
--dataset.image_transforms.enable=true  # Data augmentation
# Brightness, contrast, saturation, hue, sharpness, affine transforms
```

### Key Learnings

1. **Epochs matter more than steps**: Calculate actual epochs based on dataset size
2. **Lower loss ≠ better model**: 0.014 vs 0.012 loss, but better generalization with fewer epochs
3. **Small datasets need careful tuning**: 241 episodes vulnerable to overfitting
4. **Monitor validation metrics**: Not just training loss

**Impact**: Reduced training time from 9 hours → 3.5 hours while improving performance

---

## Challenge 4: Action Smoothness vs Reactivity Trade-off (Moderate)

### Problem Description
Robot exhibited jerky, discontinuous motion during deployment causing unnatural movement and potential safety issues.

**Symptoms:**
- Action discontinuities: 10-26° jumps between consecutive frames
- shoulder_lift: up to 26° jump (2-3x larger than training)
- elbow_flex: up to 24° jump
- wrist_flex: up to 20° jump
- Training data showed only 3-13° max jumps (much smoother)

### Technical Analysis

**v5 Configuration**:
```python
chunk_size = 10      # Predict 10 future actions (0.33 seconds)
n_action_steps = 10  # Execute all 10 before re-planning
```

**Problem**: Model re-plans every 0.33 seconds (10 frames at 30 FPS)
- Frequent re-planning → better reactivity to visual feedback
- But: Discrete planning updates → action discontinuities → jerky motion

**v4 Configuration** (smoother but less reactive):
```python
chunk_size = 30      # Predict 30 future actions (1.0 second)
n_action_steps = 30  # Execute all 30 before re-planning
```

### Solution

**Optimal balance** (v6 configuration):
```python
chunk_size = 30          # Predict 1 second ahead (smooth trajectory)
n_action_steps = 20      # Execute 0.67 seconds before re-planning
```

**Reasoning**:
- chunk_size=30: Model plans smooth 1-second trajectory
- n_action_steps=20: Re-plan every 0.67 seconds for some reactivity
- Trajectory interpolation over 30 steps → smoother motion
- Partial execution allows mid-course corrections

**Expected improvements**:
- Smoother motion (similar to v4)
- Better reactivity than v4 (re-plans 1.5x more frequently)
- Reduced action discontinuities

### Key Learnings

1. **Chunk size affects smoothness**: Larger chunks → smoother trajectories
2. **n_action_steps affects reactivity**: Smaller values → more replanning → more reactive
3. **Trade-off is fundamental**: Can't maximize both smoothness and reactivity
4. **Task-dependent tuning**: Grasping needs smoothness > reactivity (unlike dynamic tasks)

**Impact**: Improved motion quality for safer, more natural robot behavior

---

## Challenge 5: Batch Size Confusion (Resolved)

### Problem Description
Initial concern that batch_size=32 might be causing overfitting, considering reducing to 5, 8, or 10.

### Analysis

**Common misconception**: Smaller batch size prevents overfitting through gradient noise

**Mathematical reality**:
```
Overfitting ∝ Number of epochs
Epochs = (Total steps × Batch size) / Dataset size

With batch_size=32, steps=50,000:
Epochs = (50,000 × 32) / 100,832 = 15.87 

With batch_size=10, steps=50,000:
Epochs = (50,000 × 10) / 100,832 = 4.96 
BUT: Training time 3-4x longer for same epochs!
```

**Correct approach**: Reduce steps, keep batch_size=32
```
Epochs = (20,000 × 32) / 100,832 = 6.35 
Training time: 3.5 hours (optimal)
```

### Solution

**Keep batch_size=32** because:
1. Optimal GPU utilization (90-95% vs 60-75% with small batches)
2. Stable gradient estimates (not too noisy)
3. Fast training (2.5-3.5 hours vs 10-15 hours)
4. Industry standard for ~100K frame datasets
5. SmolVLA flow-matching benefits from low-noise gradients

**Control overfitting via epoch count**, not batch size.

### Key Learnings

1. **Batch size affects efficiency, not overfitting**: Overfitting controlled by epochs
2. **Don't confuse parameters**: Steps × Batch ÷ Data = Epochs (what actually matters)
3. **Smaller batch ≠ better**: Just slower training for same generalization
4. **GPU efficiency matters**: Underutilized GPUs waste time and money

**Impact**: Saved 6-12 hours per training run, avoided unnecessary experimentation

---

## Challenge 6: Starting State Inconsistency (Solved)

### Problem Description
Robot performance varied dramatically between episodes despite using same trained model.

**Observations**:
- Episode 0: Good starting state → Failed (gripper stayed open)
- Episode 1: Very different starting state (137° off) → Failed
- Episode 2: Moderate starting state → Success!
- Inconsistent success rate (0-33%) on same model

### Root Cause

**Training data characteristics**:
```
Average starting state:
  shoulder_pan: -8° ± 3°
  shoulder_lift: -98° ± 1°
  elbow_flex: 100° ± 0.1°
  gripper: 0.5° ± 0.3° (CLOSED!)
```

**Deployment inconsistency**:
- Follower arm doesn't auto-reset between episodes
- Robot stayed in random positions from previous episode
- Starting state varied wildly (0-159° from training average)
- **Gripper started OPEN (2.8-40°) instead of CLOSED (0.5°)**

### Solution

**Reset protocol** using leader arm teleoperation:
```bash
# Between EVERY episode:
1. Use leader arm to position follower to home
2. Match training starting state:
   - shoulder_pan ≈ -8°
   - shoulder_lift ≈ -98°
   - elbow_flex ≈ 100°
   - wrist_flex ≈ 75°
   - wrist_roll ≈ -52°
   - gripper ≈ 0.5° (CLOSED!) ← Critical!
3. Then start deployment episode
```

**Automated solution** (future):
```python
# Script to reset robot programmatically
from lerobot.robots.so101_follower import SO101Follower

HOME_POSITION = {
    'shoulder_pan.pos': -8.0,
    'shoulder_lift.pos': -98.0,
    'elbow_flex.pos': 100.0,
    'wrist_flex.pos': 75.0,
    'wrist_roll.pos': -52.0,
    'gripper.pos': 0.5  # CLOSED
}

robot = SO101Follower(config)
robot.connect()
robot.send_action(HOME_POSITION)
time.sleep(3)  # Wait for movement
robot.disconnect()
```

### Key Learnings

1. **Starting state matters**: Model expects specific initial configuration
2. **Gripper state critical**: Open vs closed starting position changes behavior
3. **Consistency is key**: Small variations acceptable, but need to be in same range
4. **Document home position**: Save exact starting pose with training data
5. **Automate resets**: Manual positioning prone to human error

**Impact**: Improved consistency, clearer debugging (removed starting state as variable)

---

## Challenge 7: Visual Distribution Shift (Identified)

### Problem Description
Even with correct language, camera config, and starting state, v4 model still showed 0% success rate (gripper never closed).

**Evidence**:
- Perfect starting states (close to training averages) 
- Correct language instruction ("Grab the brain") 
- Correct camera configuration 
- Still: Gripper mean 28.29 vs training 6.83 

### Analysis

**Training data quality check**:
```python
Success episodes: 20/20 (100%)
Gripper closed frames: 73.7%
Mean gripper: 7.03
All episodes showed successful grasps
```

**Conclusion**: Training data is excellent, so issue must be **visual/scene distribution shift**

**Possible causes**:
1. Camera viewpoints slightly different
2. Object placement distribution different
3. Lighting conditions changed
4. Background/table appearance different
5. Model overfitted to specific visual features of training environment

### Proposed Solution

**Mixed-dataset training strategy**:
1. Record 30-50 episodes in **actual deployment environment** using teleoperation
2. Merge with original 241-episode training dataset
3. Train v6 on combined dataset (291 episodes total)
4. Model learns to work in BOTH environments

**Advantages**:
- Learns general grasping strategy across environments
- Doesn't waste existing 241 episodes
- More robust to visual variations
- Target workspace: 2×2 inch → 6×6 inch generalization

**Implementation**:
```bash
# Step 1: Record deployment mix
lerobot-record --dataset.repo_id=deployment_mix --num_episodes=50

# Step 2: Train on merged dataset
lerobot-train --dataset.repo_id="training_data,deployment_mix" --steps=25000
```

### Key Learnings

1. **Visual consistency critical**: Even subtle environmental differences affect VLA models
2. **Overfitting to visuals**: Model can memorize camera viewpoints, not just actions
3. **Multi-environment training**: Improves generalization more than single-environment oversampling
4. **Distribution shift common**: Lab training ≠ deployment environment (always verify)

**Impact**: Developing systematic approach to visual generalization (work in progress)

---

## Phase 2 Challenges (March 2026)

---

## Challenge 8: Training Methodology Discovery (Critical)

### Problem Description
After 9 model versions and 30 Optuna trials, SmolVLA was capped at ~33% success. The initial conclusion was that SmolVLA 500M lacked the capacity for precise manipulation — a **wrong diagnosis**.

### Investigation Process

1. **Literature review**: Studied Diffusion Policy [Chi et al., 2023] iterative behavior policy refinement, SmolVLA paper ablation tables, and community fine-tuning reports
2. **Key insight from SmolVLA paper**: batch_size=64, lr=1e-4 with frozen vision encoder achieved 75-90% success. Our Phase 1 used batch=32, lr=4.24e-5 (HPO-optimized but wrong)
3. **Paper ablation**: lr=1e-5 gives **0% success rate**. Our HPO had converged toward lower LRs — exactly the wrong direction

### Solution
Complete training methodology shift:
- batch_size=64 (not 32), lr=1e-4 (not 4.24e-5)
- freeze_vision_encoder=True, train_expert_only=True
- chunk_size=50 (not 10), n_action_steps=50
- Curriculum training from checkpoints [Bengio et al., 2009]

### Key Learnings
1. **HPO can converge to wrong optima** when the search space excludes the correct configuration
2. **Paper ablations are essential reading** — the SmolVLA paper explicitly showed lr=1e-5 fails
3. The issue was training methodology, not model capacity

---

## Challenge 9: use_degrees Normalization Mismatch (Critical)

### Problem Description
After upgrading LeRobot from v0.4 to v0.5+, the robot exhibited ~5° height errors on all joints. Actions were visibly off despite using the same model.

### Investigation Process
1. Discovered LeRobot v0.5+ changed the default normalization from `RANGE_M100_100` to `DEGREES`
2. Dataset was recorded with `use_degrees=False` (RANGE_M100_100), but the new default assumed degrees
3. The normalization mismatch scaled all action values incorrectly

### Solution
Explicitly set `use_degrees=False` in both follower and leader configs:
```python
# config_so_follower.py
use_degrees: bool = False  # Must match dataset recording format
```

### Key Learnings
1. **Always verify normalization after framework updates** — replay a dataset episode to confirm
2. **Document the recording format** with the dataset
3. Breaking changes in defaults can silently corrupt trained policies

---

## Challenge 10: Garbage Episode in Dataset (Data Quality)

### Problem Description
Dataset analysis revealed episode 240 was **completely static** — 533 frames where all 6 joints had exactly zero action variance. The robot was sitting idle.

### Investigation Process
1. Ran comprehensive dataset quality audit: per-episode action statistics
2. Found episode 240 with std=0.0000 on ALL joints
3. Also discovered metadata inconsistency: `info.json` said 239 episodes, but parquet files contained 240 (indices 0-238 + 240, with 239 missing)
4. Root cause: earlier manual episode deletion removed episode 239 but left the orphaned episode 240 in `file-006.parquet`

### Solution
```bash
# Removed orphan files from local dataset + HuggingFace Hub
rm data/chunk-000/file-006.parquet
rm meta/episodes/chunk-000/file-006.parquet
# Also removed orphan video files via HuggingFace API
```

Verified: 239 episodes (0-238), 99,845 frames, no gaps.

### Key Learnings
1. **Dataset auditing is essential** — one garbage episode (0.5% of data) can measurably degrade performance
2. **Manual deletions leave orphans** — always verify index continuity after removing episodes
3. **Zero-variance data teaches "do nothing"** — directly counterproductive for manipulation tasks

---

## Challenge 11: Task Text Mismatch — Refined Understanding

### Problem Description
Same class of bug as Challenge 1, but discovered **again** in Phase 2. Dataset stored the task as "Grab the brain" but the evaluation client used "Grab the grey brain toy and place it inside the green container."

### Investigation Process
1. Checked `tasks.parquet` in dataset metadata — task text was "Grab the brain"
2. Client code had a different, more descriptive default string
3. SmolVLA creates distinct embeddings for semantically similar but textually different instructions [1]

### Solution
Changed client default to exact dataset text:
```python
task: str = "Grab the brain"  # Must match training exactly
```

### Key Learnings
1. **This bug class is persistent** — even after fixing it once, it recurred with different text
2. **VLA prompt engineering requires exact matching**, not semantic equivalence
3. **Automate prompt verification** — compare eval prompt against dataset metadata programmatically

---

## Challenge 12: Flow Matching Denoising Precision — THE Breakthrough

### Problem Description
Even with correct training config and curriculum training, the robot still showed ~50% grasp success. The model would approach the object, attempt to grasp, release, try a different angle, release again — oscillating between grasping strategies.

### Investigation Process
1. **Analyzed the behavior**: Model had learned multiple grasping approaches (one per object position in training data). At inference, it couldn't commit to a single strategy
2. **Compared ACT vs SmolVLA**: ACT (deterministic regression) grasps consistently. SmolVLA (stochastic flow matching [5]) produces slightly different actions each inference
3. **Identified root cause**: Default `num_steps=10` for flow matching denoising produced insufficient precision. The 10-step denoising process couldn't resolve the multimodal action distribution into a single, committed grasp trajectory
4. **Hypothesis**: More denoising steps should allow the model to commit to one mode. Fewer action steps before re-observation should allow faster correction

### Solution
| Parameter | Default | Optimized | Effect |
|-----------|---------|-----------|--------|
| `num_steps` | 10 | **20** | 2x denoising → precise action generation |
| `n_action_steps` | 20 | **10** | 2x re-observation → faster closed-loop correction |

### Key Learnings
1. **Inference parameters can matter as much as training** — this single change was the largest improvement in the entire project
2. **Flow matching precision scales with denoising steps** [5] — default values are not optimal for all tasks
3. **Multimodal action distributions need sufficient denoising** to resolve into committed trajectories
4. **Closed-loop frequency matters** — re-observing every 10 steps enables mid-grasp correction

**Impact**: ~50% → 85% success rate. The single most impactful discovery.

---

## Challenge 13: Batch Size and Linear Scaling Rule

### Problem Description
Attempted to improve training by increasing batch size from 64 to 128 to 200. Batch=128 improved loss (0.038 vs 0.052), but batch=200 showed diminishing returns.

### Investigation Process
1. Batch=64 (v1): loss 0.052 in 50K steps
2. Batch=128 (v2): loss 0.038 in 20K steps — significant improvement
3. Batch=200 (v3 attempt): loss plateaued, convergence slower than expected
4. **Root cause**: Linear scaling rule [6] — when batch size increases, LR should increase proportionally. We kept LR=1e-4 for all runs

### Solution
Abandoned batch=200 and returned to batch=64 with more training steps (70K). The extended training with original batch size outperformed the larger batch:
- Batch=64, 70K steps → loss **0.028** (best)
- Batch=200, early → loss ~0.040 (plateau)

### Key Learnings
1. **Linear scaling rule [6]** applies to fine-tuning but breaks at large batch sizes
2. **More steps with correct batch > fewer steps with larger batch** for this dataset size (239 episodes)
3. **Don't chase lower loss through batch size alone** — training duration matters more

---

## Challenge 14: Curriculum Training Discovery

### Problem Description
Each training run started from scratch (pretrained SmolVLA base), requiring 50K+ steps to converge. Switching to training from previous checkpoints dramatically improved convergence.

### Investigation Process
1. Observed v2 (starting from v1 checkpoint) converged to lower loss (0.038) in only 20K steps
2. v3 (starting from v2 checkpoint) achieved the lowest loss (0.028) in 70K steps
3. This follows the principle of curriculum learning [7] — start with a broadly trained policy and progressively refine

### Solution
Curriculum training pipeline:
```
SmolVLA base → Proven v1 (50K steps) → Proven v2 (20K steps) → Proven v3 (70K steps)
```

Each iteration inherits the learned representations and refines them further, similar to iterative policy refinement in Diffusion Policy [2].

### Key Learnings
1. **Curriculum training [7] outperforms training from scratch** for small datasets
2. **Progressive checkpoint refinement** enables exploring different batch sizes and schedules
3. **Each iteration can focus on different aspects** — v1 learns basics, v2 refines with larger batch, v3 extends with more steps

---

## Challenge 15: Remote Inference Architecture

### Problem Description
Local GPU (RTX 3050 Laptop) was too slow for real-time SmolVLA inference. The model requires ~200ms per inference on an RTX 4090 but takes 2-3 seconds on RTX 3050.

### Investigation Process
1. Profiled inference latency: local GPU ~2.5s per chunk (unacceptable at 10 FPS control)
2. Cloud GPU (RunPod) achieves ~200ms but robot hardware is local
3. Needed to split inference (cloud) from control (local)

### Solution
Designed server-client architecture:
- **Server** (`smolvla_server.py`): FastAPI HTTP server on RunPod GPU, loads model once, serves inference via JSON
- **Client** (`smolvla_client.py`): Multi-threaded — inference requests, action execution, video recording
- **Connection**: SSH tunnel for secure communication
- **Latency**: ~200-300ms round-trip including network

### Key Learnings
1. **Server-client split enables GPU-limited researchers** to use powerful models with local robots
2. **SSH tunneling** provides secure communication without exposing ports
3. **Multi-threaded client** prevents inference latency from blocking motor control

---

## Challenge 16: Video Recording vs Robot Performance

### Problem Description
Adding camera recording to the eval client degraded robot performance. The robot moved slower and less precisely when recording was active.

### Investigation Process
1. Recording thread called `robot.get_observation()` with `robot_lock` — competing with inference thread
2. Each camera read takes ~50-100ms, doubling the lock contention
3. The inference thread was delayed, causing stale action predictions

### Solution
Shared frame buffer approach:
- Inference thread writes camera frames to shared buffer after each observation
- Recorder thread reads from buffer without acquiring robot lock
- Zero additional camera reads, zero lock contention
- Used pyav (h264) encoder for compatible video output with signal handler for graceful shutdown

### Key Learnings
1. **Resource contention** between threads can silently degrade real-time control
2. **Shared memory patterns** enable recording without performance impact
3. **Graceful shutdown** is essential for video encoding — SIGTERM must flush the encoder

---

## Challenge 17: Disk Space Management (RunPod + Local)

### Problem Description
Both RunPod (30GB disk + 30GB volume) and local machine (53GB, 1GB free) ran critically low on storage during training.

### Investigation Process
1. **RunPod**: 10 checkpoints x 1.3GB = 13GB, HF cache 7.9GB, wandb cache 12GB
2. **Local**: Firefox cache 3.9GB, snap old revisions 5.5GB, VS Code 2.3GB

### Solution
- Deleted intermediate checkpoints (kept only key ones: 20K, 50K)
- Cleaned wandb cache (12GB freed)
- Cleaned HF model cache for unused models
- Locally: purged Firefox cache, snap old revisions
- Result: RunPod root 20% used (from 67%), local 3.5GB free (from 1GB)

### Key Learnings
1. **Monitor disk space during long training runs** — 50K steps generate many checkpoints
2. **Clean wandb cache periodically** — it grows silently
3. **Only keep essential checkpoints** — final model is pushed to HuggingFace Hub

---

## Summary of Impact

### Phase 1

| Challenge | Time Lost | Solution Time | Impact |
|-----------|-----------|---------------|--------|
| Language mismatch | 2 weeks | 2 days | Critical — 0% → 33% success |
| Camera swap | 1 week | 1 day | Critical — Prevented all progress |
| Overfitting | 1 week | 3 days | High — 2.5x faster training |
| Action smoothness | 3 days | 1 day | Moderate — Safer motion |
| Batch size confusion | 2 days | 4 hours | Low — Avoided wasted effort |
| Starting state | 2 days | 1 day | Moderate — Improved consistency |
| Visual distribution | 3 days | 2 days | High — Key to generalization |

### Phase 2

| Challenge | Time Lost | Solution Time | Impact |
|-----------|-----------|---------------|--------|
| Training methodology | 4 weeks | 3 days | **Critical — 33% → 50%** |
| use_degrees mismatch | 2 days | 2 hours | Critical — Prevented deployment |
| Garbage episode | 1 day | 2 hours | Moderate — Cleaner training |
| Task text (refined) | 1 day | 1 hour | Moderate — Improved confidence |
| **Denoising precision** | **3 days** | **4 hours** | **Critical — 50% → 85%** |
| Batch size scaling | 2 days | 1 day | Moderate — Optimal convergence |
| Curriculum training | 1 day | 1 day | High — Faster convergence |
| Remote inference | 3 days | 2 days | High — Enabled deployment |
| Recording contention | 1 day | 4 hours | Low — Clean video capture |
| Disk management | 1 day | 2 hours | Low — Prevented interruptions |

**Total debugging time**: ~12 weeks across both phases

**Skills demonstrated**:
- Systematic debugging methodology
- Root cause analysis with quantitative evidence
- Hypothesis-driven experimentation
- Deep understanding of VLA model internals (flow matching, denoising, language conditioning)
- Data-centric approach to ML (dataset auditing, quality verification)
- Literature-informed research (Diffusion Policy, Curriculum Learning, Linear Scaling Rule)
- Systems engineering (server-client architecture, multi-threaded control)
- Persistence across 17 distinct technical challenges

---

## Key Takeaways for Future Projects

### 1. Documentation is Critical
- Document exact training configuration (language, camera setup, starting state)
- Save environment details with model checkpoints
- Version everything (data, code, configs)

### 2. Verify Assumptions Early
- Check dataset metadata vs deployment config
- Visually verify camera feeds
- Test on simple cases before complex scenarios

### 3. Understand Model Architecture
- Know how VLAs process language (exact matching required)
- Understand visual feature extraction (camera consistency)
- Learn action prediction pipeline (normalization, denormalization)

### 4. Start Simple, Add Complexity
- Fix one variable at a time
- Isolate issues systematically
- Don't change multiple things simultaneously

### 5. Metrics Tell Stories
- Analyze action distributions (gripper 54.5% → 0%)
- Compare training vs deployment statistics
- Use quantitative evidence for debugging

---

## For Resume/Interview

**When asked "Tell me about a challenging problem you solved":**

*"During my robot learning project, I encountered a critical bug where the trained model achieved 0% success rate in deployment despite 100% success in training. Through systematic debugging, I discovered three interconnected issues:*

1. *Language instruction mismatch - VLA models require exact text matching*
2. *Camera configuration swap - spatial reasoning completely inverted*
3. *Visual distribution shift - model overfitted to training environment*

*I developed diagnostic tools to analyze action distributions, verified hardware configurations, and implemented a mixed-dataset training strategy. This improved success rate from 0% to 33% and led to a systematic approach for preventing similar issues in future deployments."*

**Demonstrates**: Problem-solving, debugging skills, systematic thinking, persistence, technical depth
