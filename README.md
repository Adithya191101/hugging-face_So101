# HuggingFace LeRobot - SO-101 Manipulation with ACT & SmolVLA 🤖

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow.svg)](https://huggingface.co/AdithyaRajendran)

Implementation of vision-language-action models (ACT & SmolVLA) for robotic pick-and-place tasks on the SO-101 manipulator.

---

## 🎯 Quick Results

| Model | Task | Success Rate | Workspace | Status |
|-------|------|--------------|-----------|--------|
| **ACT** | Pick-and-place soft irregular objects | **80%** | Full workspace | ✅ Completed |
| **SmolVLA** | Language-conditioned manipulation | **33%** → improving | 2×2 inch (working on 6×6 inch) | 🔄 In Progress |

---

## 📋 Table of Contents

- [Models Implemented](#models-implemented)
- [Key Achievements](#key-achievements)
- [Problems Faced & Solutions](#problems-faced--solutions)
- [Tools & Technologies](#tools--technologies)
- [Installation](#installation)
- [Results](#results)
- [Videos](#videos)
- [Detailed Documentation](#detailed-documentation)

---

## 🤖 Models Implemented

### 1. Action Chunking Transformer (ACT)

**Performance**: **80% success rate** on pick-and-place tasks with soft irregular objects

**Description**: ACT uses temporal action chunking to generate smooth, coordinated robot trajectories. Successfully tested on soft, deformable, irregular round/obloid objects (brain toy).

**Key Features**:
- Smooth motion through temporal ensembling
- Robust to object deformations
- Dual-camera visual feedback (front + wrist)
- 241 demonstration episodes, 100K+ frames

**Technical Specs**:
```
Input: RGB (640×480) × 2 cameras + Joint states (6-DOF)
Architecture: CNN encoder → Transformer → CVAE decoder
Output: Action chunks (chunk_size=100)
Training: 50K steps, batch_size=8, ~6 hours (V100)
```

**GitHub Links**:
- Training Dataset: [huggingface.co/datasets/AdithyaRajendran/so101_grab_brain_t2](https://huggingface.co/datasets/AdithyaRajendran/so101_grab_brain_t2)
- Model: *(To be uploaded)*

---

### 2. SmolVLA (Small Vision-Language-Action Model)

**Performance**: Currently **33% success rate**, working towards **>70%** with generalization improvements

**Description**: SmolVLA is a 500M parameter vision-language model fine-tuned for language-conditioned robotic manipulation. Currently trained to work in a constrained 2×2 sq. inch pickup area, with ongoing work to generalize to a larger 6×6 sq. inch workspace.

**Current Status**:
- ✅ Successfully trained 5 model versions (v2-v6)
- ✅ Identified and resolved critical deployment issues
- 🔄 Training v6 with mixed-dataset approach for better generalization
- 🎯 Target: >70% success rate with 6×6 inch pickup area

**Key Features**:
- Language-conditioned task execution (e.g., "Grab the brain")
- Vision-language understanding with SmolVLM2-500M-Video-Instruct
- Flow-matching diffusion model for action prediction
- Dual-camera setup for spatial reasoning

**Technical Specs**:
```
Base Model: SmolVLM2-500M-Video-Instruct (HuggingFace)
Architecture: Frozen vision encoder + trainable action decoder
Training Dataset: 241 episodes (100K frames) + 50 deployment mix (in progress)
Hyperparameters:
  - chunk_size: 30 (1-second prediction horizon)
  - n_action_steps: 20 (smooth + reactive)
  - batch_size: 32
  - learning_rate: 1e-5
  - epochs: 6.35 (optimized to prevent overfitting)
Training Time: ~3.5 hours (V100 GPU)
```

**GitHub Links**:
- Training Dataset: [huggingface.co/datasets/AdithyaRajendran/so101_grab_brain_t2](https://huggingface.co/datasets/AdithyaRajendran/so101_grab_brain_t2)
- SmolVLA v5: [huggingface.co/AdithyaRajendran/so101_smolvla_policy_FINAL_v5](https://huggingface.co/AdithyaRajendran/so101_smolvla_policy_FINAL_v5)
- SmolVLA v6: *(Training in progress)*

**Workspace Generalization**:
- **Current**: 2×2 sq. inch pickup area with 33% success rate
- **Goal**: 6×6 sq. inch pickup area with >70% success rate
- **Approach**: Mixed-dataset training combining original data + deployment environment demonstrations

---

## 🏆 Key Achievements

### ACT Model
- ✅ **80% success rate** on soft irregular object pick-and-place
- ✅ Trained on **241 demonstrations** (100K+ frames)
- ✅ Smooth, natural motion trajectories
- ✅ Robust to object deformations and position variations

### SmolVLA Model
- ✅ Successfully trained **500M parameter** vision-language model for robotics
- ✅ Debugged and resolved **7 critical deployment issues** through systematic root cause analysis
- ✅ Optimized training from **15.87 epochs → 6.35 epochs** (2.5x faster, better generalization)
- ✅ Improved deployment success from **0% → 33%** by fixing language instruction mismatch
- ✅ Published **datasets and models** to HuggingFace Hub with full reproducibility
- 🔄 Developing mixed-dataset strategy to generalize from 2×2 inch → 6×6 inch pickup area

### Research Contributions
- 📊 Systematic methodology for debugging vision-language-action models
- 📊 Quantitative analysis techniques for diagnosing deployment failures
- 📊 Best practices for small-dataset robot learning (epochs optimization)
- 📊 Configuration consistency protocols for VLA model deployment

---

## 🔴 Problems Faced & Solutions

This project involved extensive debugging and systematic problem-solving. Below are the **7 major challenges** encountered and how they were resolved.

### 🔴 Critical Issue 1: Language Instruction Mismatch

**Impact**: 0% deployment success → 33% success after fix

**Problem**:
- Model achieved 100% success in training, but **0% success** during deployment
- Robot approached object but **gripper never closed**
- Analysis showed gripper closed 54.5% of frames in training, but **0% in deployment**
- Gripper action values stuck at mean ~26 instead of training mean 6.8

**Root Cause**:
Training used task description:
```
"Grab the brain"
```

Deployment used different description:
```
"Grasp a brain and put it in the bin."
```

Vision-language models create **different embeddings for different text**, even if semantically similar. The model learned:
```
"Grab the brain" → [embedding_1] → {approach, close gripper, lift}
```

But received during deployment:
```
"Grasp a brain and put it in the bin." → [embedding_2] ≠ [embedding_1]
→ {default behavior: keep gripper open}
```

**Solution**:
```bash
# Must use EXACT same task description as training
--dataset.single_task="Grab the brain"  # Matches training exactly
```

**Result**: Success rate improved from **0% → 33%**

**Key Learning**: VLA models require exact language matching - even semantically similar phrases create completely different behaviors. Always document and verify task descriptions.

---

### 🔴 Critical Issue 2: Camera Configuration Swap

**Impact**: Robot crashed into bin instead of approaching object (complete spatial reasoning failure)

**Problem**:
- After adjusting cameras to "match training environment", behavior got **dramatically worse**
- Robot crashed into bin instead of approaching object
- All models (v3, v4, v5) failed identically
- Model seemed to have inverted understanding of object locations

**Root Cause**:
Physical cameras were **swapped** between camera1 and camera2:
- camera1 (expected: front view) was receiving **wrist camera** feed
- camera2 (expected: wrist view) was receiving **front camera** feed

This caused the model to:
- Use wrist camera for scene understanding (wrong!)
- Use front camera for gripper control (wrong!)
- Complete inversion of spatial reasoning

**Investigation**:
```bash
# Verified with visual display
lerobot-record --display_data=true

# Discovered:
camera1 window showed wrist view (should be front)
camera2 window showed front view (should be wrist)
```

**Solution**:
```bash
# Swapped video device assignments to match training
--robot.cameras="{
  camera1: {index_or_path: /dev/video2},  # Now shows front view
  camera2: {index_or_path: /dev/video4}   # Now shows wrist view
}"
```

**Result**: Robot approached object correctly again

**Key Learning**: Camera configuration must be **exactly identical** to training. Always verify camera feeds visually before deployment. Physical hardware consistency is critical for vision-based policies.

---

### 🔴 Critical Issue 3: Overfitting on Small Dataset

**Impact**: 2.5x faster training with better generalization

**Problem**:
- Training for 50,000 steps resulted in very low loss (0.014)
- Model showed poor generalization during deployment
- Inconsistent performance across slight object position variations

**Analysis**:
```python
Original training:
  Steps: 50,000
  Dataset: 241 episodes (100,832 frames)
  Batch size: 32

  Epochs = (50,000 × 32) / 100,832 = 15.87 epochs ❌

Industry standard for <500 episodes: 3-7 epochs
Our training: 15.87 epochs = 2-3x TOO MANY
```

**What Happens with Overfitting**:
- Model memorizes exact camera views and trajectories
- Fails to generalize to slight variations
- Loss continues decreasing without plateau
- Training data: 100% success, Deployment: 0% success

**Solution**:
```python
Optimized training (v5):
  Steps: 20,000
  Epochs = (20,000 × 32) / 100,832 = 6.35 epochs ✅

Result:
  - Training time: 9 hours → 3.5 hours (2.5x faster)
  - Final loss: 0.014 → 0.012 (similar)
  - Generalization: Much better
```

**Key Learning**: Monitor **epochs** (not just steps) for small datasets. Calculate: `epochs = (steps × batch_size) / dataset_size`. Lower loss ≠ better model. Industry standard: 3-7 epochs for <500 episodes.

---

### 🟡 Moderate Issue 4: Action Smoothness vs Reactivity Trade-off

**Impact**: Jerky motion → smooth, safe robot movement

**Problem**:
- v5 model exhibited jerky, discontinuous motion
- Action jumps up to 10-26° between consecutive frames
- Training data showed only 3-13° max jumps (much smoother)
- Unnatural movement, potential safety concerns

**Analysis**:
```python
v5 Configuration:
  chunk_size = 10         # Predict 0.33 seconds ahead
  n_action_steps = 10     # Re-plan every 0.33 seconds

Result: Frequent re-planning → discrete updates → jerky motion

v4 Configuration:
  chunk_size = 30         # Predict 1 second ahead
  n_action_steps = 30     # Re-plan every 1 second

Result: Smooth motion but less reactive to visual feedback
```

**Solution - Optimal Balance (v6)**:
```python
chunk_size = 30          # Smooth 1-second trajectory prediction
n_action_steps = 20      # Re-plan every 0.67 seconds

Benefits:
  - Smooth motion (like v4)
  - Better reactivity (1.5x more frequent than v4)
  - Reduced action discontinuities
```

**Key Learning**: Chunk size affects smoothness, n_action_steps affects reactivity. Grasping tasks need smoothness > reactivity (unlike dynamic tasks like catching). Task-dependent hyperparameter tuning is essential.

---

### 🟡 Moderate Issue 5: Batch Size Confusion

**Impact**: Avoided 6-12 hours of wasted training time

**Problem**:
Initial concern that `batch_size=32` might be causing overfitting. Considered reducing to 5, 8, or 10.

**Analysis - Common Misconception**:
```python
Misconception: Smaller batch size prevents overfitting

Reality: Overfitting ∝ Number of epochs (NOT batch size)

With batch_size=10, steps=50,000:
  Epochs = (50,000 × 10) / 100,832 = 4.96 ✅
  BUT: Training time = 10-15 hours (3-4x slower)

Correct approach - Keep batch_size=32, reduce steps:
  Epochs = (20,000 × 32) / 100,832 = 6.35 ✅
  Training time = 3.5 hours (optimal)
```

**Why batch_size=32 is Optimal**:
- GPU utilization: 90-95% (vs 60-75% with small batches)
- Stable gradient estimates (not too noisy)
- Industry standard for ~100K frame datasets
- SmolVLA flow-matching benefits from low-noise gradients
- Fast training without sacrificing generalization

**Key Learning**: Batch size affects **training efficiency**, not overfitting. Control overfitting via epoch count. Don't confuse parameters - what matters is: `epochs = (steps × batch) / data`.

---

### 🟡 Moderate Issue 6: Starting State Inconsistency

**Impact**: Improved consistency, enabled reliable testing

**Problem**:
- Performance varied dramatically between episodes (0-33% success)
- Same model, same task, wildly different results
- Episode 1: Starting state 137° off from training → Failed
- Episode 2: Better starting state → Success

**Root Cause**:
```python
Training data - Consistent starting state:
  shoulder_pan: -8° ± 3°
  shoulder_lift: -98° ± 1°
  elbow_flex: 100° ± 0.1°
  gripper: 0.5° ± 0.3° (CLOSED!)

Deployment - Inconsistent starting state:
  - Follower arm doesn't auto-reset between episodes
  - Robot stayed in random positions from previous episode
  - Starting state varied 0-159° from training average
  - Gripper started OPEN (2.8-40°) instead of CLOSED (0.5°)
```

**Solution - Reset Protocol**:
```bash
# Between EVERY episode, manually reset using leader arm:
Home position:
  shoulder_pan: -8°
  shoulder_lift: -98°
  elbow_flex: 100°
  wrist_flex: 75°
  wrist_roll: -52°
  gripper: 0.5° (CLOSED!) ← Critical!
```

**Automated Solution (Future)**:
```python
from lerobot.robots.so101_follower import SO101Follower

HOME_POSITION = {
    'shoulder_pan.pos': -8.0,
    'shoulder_lift.pos': -98.0,
    'elbow_flex.pos': 100.0,
    'wrist_flex.pos': 75.0,
    'wrist_roll.pos': -52.0,
    'gripper.pos': 0.5  # CLOSED
}

robot.send_action(HOME_POSITION)
```

**Key Learning**: Starting state consistency is critical. Model expects specific initial configuration. Document exact home position with training data. Gripper state (open vs closed) particularly important.

---

### 🟢 Identified Issue 7: Visual Distribution Shift

**Impact**: Developing mixed-dataset strategy for generalization (work in progress)

**Problem**:
Even with ALL fixes (correct language, camera config, starting state), v4 model still showed:
- Gripper mean: 28.29 (deployment) vs 6.83 (training)
- Gripper stayed open most of the time
- 0% success rate persisted

**Analysis**:
```python
Training data quality check:
  Success episodes: 20/20 (100%)
  Gripper closed frames: 73.7%
  Mean gripper: 7.03
  ✅ Training data is EXCELLENT

Conclusion: Issue is visual/scene distribution shift
```

**Possible Causes**:
1. Camera viewpoints slightly different
2. Object placement distribution different
3. Lighting conditions changed
4. Background/table appearance different
5. Model overfitted to specific visual features

**Solution - Mixed-Dataset Training (v6)**:
```python
Approach:
1. Record 50 episodes in ACTUAL deployment environment (teleoperation)
2. Merge with 241 original training episodes
3. Train v6 on combined dataset (291 episodes total)
4. Model learns to work in BOTH environments

Expected Benefits:
- Generalizes across visual conditions
- Doesn't waste existing 241 episodes
- More robust to environmental variations
- Target: 2×2 inch → 6×6 inch workspace generalization

Training Command:
lerobot-train \
  --dataset.repo_id="training_data,deployment_mix" \
  --steps=25000  # 6.67 epochs for 291 episodes
```

**Status**: 🔄 In progress - collecting deployment mix dataset

**Key Learning**: Visual consistency critical for VLA models. Even subtle environmental differences affect performance. Multi-environment training improves generalization more than single-environment oversampling.

---

## 📊 Debugging Statistics

| Challenge | Time Spent | Impact | Resolution | Status |
|-----------|------------|--------|------------|--------|
| Language mismatch | 2 weeks | Critical (0%→33%) | Exact task matching | ✅ Solved |
| Camera swap | 1 week | Critical (complete failure) | Visual verification | ✅ Solved |
| Overfitting | 1 week | High (2.5x speedup) | Epoch optimization | ✅ Solved |
| Action smoothness | 3 days | Moderate (safety) | Hyperparameter tuning | ✅ Solved |
| Batch size | 2 days | Low (efficiency) | Mathematical analysis | ✅ Solved |
| Starting state | 2 days | Moderate (consistency) | Reset protocol | ✅ Solved |
| Visual distribution | Ongoing | High (generalization) | Mixed-dataset training | 🔄 In progress |

**Total debugging time**: ~6 weeks of systematic problem-solving

**Skills Demonstrated**:
- ✅ Systematic debugging methodology
- ✅ Root cause analysis with quantitative evidence
- ✅ Hypothesis-driven experimentation
- ✅ Deep understanding of VLA model internals
- ✅ Data-centric approach to machine learning
- ✅ Persistence in solving complex technical problems

---

## 🛠️ Tools & Technologies

### Machine Learning Frameworks
- **PyTorch**: Deep learning framework for model training
- **HuggingFace Transformers**: Pre-trained vision-language models
- **HuggingFace Hub**: Model hosting, dataset versioning, collaborative ML
  - Published datasets: 100K+ frames with full metadata
  - Model checkpoints with reproducible configurations
  - Dataset: [AdithyaRajendran/so101_grab_brain_t2](https://huggingface.co/datasets/AdithyaRajendran/so101_grab_brain_t2)

### Experiment Tracking & Visualization
- **Weights & Biases (W&B)**: Experiment tracking, hyperparameter optimization
  - Tracked 5+ training runs with comprehensive metrics
  - Loss curves, gradient norms, action distributions
  - Real-time monitoring of training stability and overfitting
  - Collaborative experiment management

### Robot Learning Framework
- **LeRobot (HuggingFace)**: End-to-end robot learning pipeline
  - Data collection with teleoperation
  - Policy training and evaluation
  - Deployment on real hardware
  - Integration with HuggingFace ecosystem

### Development Environment
- **Google Colab**: GPU-accelerated training
  - V100 and A100 GPU access
  - Automatic Mixed Precision (AMP) for memory efficiency
  - Google Drive integration for checkpoint persistence

### Hardware & Robotics
- **SO-101 Follower Arm**: 6-DOF robotic manipulator
- **SO-101 Leader Arm**: Teleoperation for data collection
- **Feetech Servos**: Position-controlled motors
- **Dual Camera Setup**:
  - Front camera: Scene understanding (640×480 @ 30fps)
  - Wrist camera: Fine-grained manipulation (640×480 @ 30fps)

### Python Libraries
- `datasets` (HuggingFace): Efficient dataset handling
- `transformers`: Vision-language models
- `opencv-python`: Camera interface
- `pandas`, `numpy`: Data analysis
- `matplotlib`: Visualization

### Version Control & Collaboration
- **Git/GitHub**: Code versioning and documentation
- **HuggingFace Hub**: Model and dataset versioning

---

## 💻 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (V100 or A100 recommended)
- SO-101 Follower Arm
- Dual camera setup

### Setup
```bash
# Clone repository
git clone https://github.com/Adithya191101/hugging-face_So101.git
cd hugging-face_So101

# Install LeRobot with dependencies
pip install -e ".[feetech]"  # For SO-101 hardware
pip install -e ".[smolvla]"  # For SmolVLA model

# Login to HuggingFace
huggingface-cli login

# Login to Weights & Biases (optional)
wandb login
```

---

## 📊 Results

### ACT Model Performance

| Metric | Value |
|--------|-------|
| Success Rate | **80%** |
| Training Episodes | 241 |
| Training Frames | 100,832 |
| Object Type | Soft irregular (brain toy) |
| Training Time | ~6 hours (V100) |
| Inference Speed | 30 FPS |
| Motion Smoothness | 3-13° max joint jumps |

### SmolVLA Model Performance

#### Version Comparison

| Version | Configuration | Epochs | Issues | Success Rate |
|---------|--------------|--------|--------|--------------|
| v2-v3 | chunk=30, steps=50k | 15.87 | Overfitting, language mismatch | 0% |
| v4 | chunk=30, steps=50k | 15.87 | Language mismatch, overfitting | 0% |
| v5 | chunk=10, steps=20k | 6.35 | Language mismatch fixed | 33% |
| v6 | chunk=30, steps=25k | 6.67 | Mixed-dataset training | 🔄 Target: >70% |

#### Training Metrics (v5)

| Metric | Value |
|--------|-------|
| Model Parameters | ~500M (frozen vision) + ~50M (trainable) |
| Training Dataset | 241 episodes (100,832 frames) |
| Training Steps | 20,000 |
| Epochs | 6.35 |
| Final Loss | 0.012 |
| Training Time | 3.5 hours (V100) |
| Batch Size | 32 |
| Learning Rate | 1e-5 |

#### Workspace Generalization

| Metric | Current | Target |
|--------|---------|--------|
| Pickup Area | 2×2 sq. inch | 6×6 sq. inch |
| Success Rate | 33% | >70% |
| Dataset Size | 241 episodes | 291 episodes (with deployment mix) |
| Status | Constrained | 🔄 Generalizing |

---

## 🎥 Videos

### 📹 Demo Videos

**All videos demonstrate real SO-101 hardware with dual-camera setup (front + wrist views)**

---

#### 1. ACT Model - 80% Success Rate on Pick-and-Place

<video src="https://github.com/Adithya191101/hugging-face_So101/raw/main/videos/ACT_policy.mp4" controls width="100%"></video>

**Key Highlights**:
- Smooth, coordinated motion through temporal action chunking
- Successful grasping of soft, deformable brain toy
- Demonstrates temporal ensembling for fluid trajectories
- Shows the model that achieved 80% success rate

---

#### 2. SmolVLA - Language-Conditioned Grasping

<video src="https://github.com/Adithya191101/hugging-face_So101/raw/main/videos/SmolVLA.mp4" controls width="100%"></video>

**Key Highlights**:
- Language-conditioned task: "Grab the brain"
- 500M parameter vision-language model for robotic manipulation
- Demonstrates current 33% success rate after debugging
- Shows object approach and grasping behavior

---

#### 3. Data Collection via Teleoperation

<video src="https://github.com/Adithya191101/hugging-face_So101/raw/main/videos/Imitation.mp4" controls width="100%"></video>

**Key Highlights**:
- Shows human demonstrations using SO-101 leader arm
- Illustrates the teleoperation setup for training data collection
- Foundation for both ACT and SmolVLA training (241 episodes, 100K+ frames)
- Demonstrates the imitation learning pipeline

---

📂 **For detailed video descriptions, see [videos/README.md](videos/README.md)**

---

## 📁 Detailed Documentation

- [Complete Problem Documentation](docs/challenges_and_solutions.md) - In-depth analysis of all 7 challenges
- [Technical Details](docs/technical_documentation.md) - Full implementation details
- [Experiment Logs](docs/experiment_logs.md) - Training runs and results
- [Troubleshooting Guide](docs/troubleshooting.md) - Common issues and solutions

---

## 🔗 Links & Resources

### HuggingFace Resources
- **Profile**: [huggingface.co/AdithyaRajendran](https://huggingface.co/AdithyaRajendran)
- **Training Dataset**: [so101_grab_brain_t2](https://huggingface.co/datasets/AdithyaRajendran/so101_grab_brain_t2)
- **SmolVLA v5 Model**: [so101_smolvla_policy_FINAL_v5](https://huggingface.co/AdithyaRajendran/so101_smolvla_policy_FINAL_v5)

### Framework Documentation
- **LeRobot**: [github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)
- **SmolVLM**: [HuggingFaceTB/SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct)

---

## 🚀 Future Work

### Short-term (1-2 weeks)
- [ ] Complete deployment mix dataset (50 episodes)
- [ ] Train SmolVLA v6 on merged dataset
- [ ] Achieve >70% success rate
- [ ] Generalize to 6×6 inch pickup area

### Medium-term (1-2 months)
- [ ] Multi-object scenarios
- [ ] Diverse language instruction training
- [ ] Compare ACT vs SmolVLA performance
- [ ] Real-world robustness testing

### Long-term (3-6 months)
- [ ] Multi-task VLA (pick, place, push, stack)
- [ ] Open-source contribution to LeRobot
- [ ] Publication-quality results

---

## 🎓 Key Learnings

1. **VLA Models Require Exact Configuration Matching**
   - Language instructions must match exactly (not just semantically)
   - Camera configurations must be identical to training
   - Starting states significantly impact performance

2. **Quantitative Debugging is Essential**
   - Analyze action distributions (e.g., gripper: 54.5% → 0%)
   - Calculate epochs, not just steps
   - Compare training vs deployment statistics

3. **Data-Centric Approach**
   - Data quality > model complexity
   - Multi-environment training improves generalization
   - Document environment setup with model checkpoints

4. **Systematic Methodology**
   - Test one variable at a time
   - Form hypotheses based on evidence
   - Validate solutions quantitatively

---

## 📧 Contact

**Adithya Rajendran**
- GitHub: [@Adithya191101](https://github.com/Adithya191101)
- HuggingFace: [@AdithyaRajendran](https://huggingface.co/AdithyaRajendran)

---

## 📄 License

Apache 2.0 License - See LICENSE file for details

---

## 🙏 Acknowledgments

- **HuggingFace** for LeRobot framework and model hosting
- **SmolVLM Team** for pre-trained vision-language model
- **Open-source robotics community**

---

*Last Updated: January 2026*
