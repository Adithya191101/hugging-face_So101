# HuggingFace LeRobot - SO-101 Manipulation with ACT & SmolVLA

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow.svg)](https://huggingface.co/AdithyaRajendran)

Implementation of vision-language-action models (ACT & SmolVLA) for robotic pick-and-place tasks on the SO-101 manipulator, with systematic hyperparameter optimization and diagnostics.

---

## Quick Results

| Model | Task | Success Rate | Status |
|-------|------|--------------|--------|
| **ACT** | Pick-and-place soft irregular objects | **80%** | Completed |
| **SmolVLA** (9 versions) | Language-conditioned manipulation | **~33% max** | Diagnosed — gripper oscillation |
| **Pi0.5** (next) | Language-conditioned manipulation | TBD | Planned |

---

## Table of Contents

- [Models Implemented](#models-implemented)
- [SmolVLA: 9 Versions and What We Learned](#smolvla-9-versions-and-what-we-learned)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Gripper Oscillation Diagnosis](#gripper-oscillation-diagnosis)
- [Problems Faced & Solutions](#problems-faced--solutions)
- [Scripts & Notebooks](#scripts--notebooks)
- [Tools & Technologies](#tools--technologies)
- [Installation](#installation)
- [Future Work](#future-work)

---

## Models Implemented

### 1. Action Chunking Transformer (ACT)

**Performance**: **80% success rate** on pick-and-place tasks with soft irregular objects

ACT uses temporal action chunking to generate smooth, coordinated robot trajectories. Successfully tested on soft, deformable, irregular round/obloid objects (brain toy).

```
Input: RGB (640x480) x 2 cameras + Joint states (6-DOF)
Architecture: CNN encoder -> Transformer -> CVAE decoder
Output: Action chunks (chunk_size=100)
Training: 50K steps, batch_size=8, ~6 hours (V100)
```

### 2. SmolVLA (Small Vision-Language-Action Model)

**Performance**: ~33% success rate across all 9 trained versions

SmolVLA is a 500M parameter vision-language model (based on SmolVLM2-500M-Video-Instruct) fine-tuned for language-conditioned robotic manipulation.

```
Base Model: SmolVLM2-500M-Video-Instruct (HuggingFace)
Architecture: Frozen vision encoder + trainable action expert
Dataset: 239 episodes (99,845 frames) — 2 bad episodes removed
Task: "Grab the grey brain toy and place it inside the green container"
Cameras: Front + Wrist (640x480 @ 30fps)
```

**Key finding**: After training 9 versions with widely varying hyperparameters, all models exhibit the same gripper oscillation behavior. Diagnosis confirmed this is a **model capacity limitation**, not a hyperparameter issue. See [Gripper Oscillation Diagnosis](#gripper-oscillation-diagnosis).

---

## SmolVLA: 9 Versions and What We Learned

### Version History

| Version | Repo ID | Chunk Size | N Action Steps | LR | Weight Decay | Key Change |
|---------|---------|-----------|---------------|------|-------------|------------|
| v1 | `so101_smolvla_policy` | 50 | 50 | 1e-4 | 1e-10 | Baseline |
| v2 | `so101_smolvla_policy_V2` | 50 | 50 | 1e-4 | 1e-10 | More training |
| v3 | `so101_smolvla_policy_FINAL` | 30 | 30 | 1e-4 | 1e-10 | Reduced chunk |
| v4 | `so101_smolvla_policy_FINAL_v2` | 15 | 10 | 1e-4 | 1e-10 | Smaller chunks |
| v5 | `so101_smolvla_policy_FINAL_v3` | 30 | 30 | 1e-4 | 1e-10 | Epoch optimization |
| v6 | `so101_smolvla_policy_FINAL_v4` | 30 | 30 | 1e-4 | 1e-10 | More augmentation |
| v7 | `so101_smolvla_policy_FINAL_v5` | 10 | 10 | 1e-4 | 1e-10 | Small chunks |
| v8 | `smolvla_so101_grab_brain_t2_v8` | 10 | 10 | 4.24e-5 | 8.49e-5 | HPO-optimized |
| v9 | `smolvla_so101_grab_brain_t2_v9` | 10 | 10 | 4.24e-5 | 8.49e-5 | HPO + cleaned dataset |

All models hosted at [huggingface.co/AdithyaRajendran](https://huggingface.co/AdithyaRajendran)

### Key Observation

**All 9 models show identical behavior**: the robot approaches the object correctly (visual understanding works, adjusts to object position) but gets jittery near the grasp point, fails to close the gripper precisely, and sometimes grabs but drops during transport. This consistency across very different hyperparameters confirms the issue is **not** hyperparameter-related.

---

## Hyperparameter Optimization

Ran **Optuna HPO with 30 trials** (15 completed, 15 pruned) to find optimal SmolVLA hyperparameters.

### Best Parameters (Trial #11, avg final loss = 0.02300)

| Parameter | Default | HPO-Optimized |
|-----------|---------|---------------|
| Learning rate | 1e-4 | **4.24e-5** (2.4x lower) |
| Weight decay | 1e-10 | **8.49e-5** (much more regularization) |
| Chunk size | varied | **10** |
| N action steps | varied | **10** |
| Freeze vision | True | **True** (confirmed) |
| Affine degrees | 0 | **9.94** |
| Affine translate | 0 | **0.107** |
| Color jitter brightness | 0 | **0.119** |
| Color jitter contrast | 0 | **0.399** |

### Training Config (v9 — best model)

```
Dataset: 239 episodes, 99,845 frames (episodes 69 & 240 deleted)
Steps: 20,290 (6.5 epochs)
Batch size: 32
Normalization: MEAN_STD (action + state), IDENTITY (visual)
Warmup: 1000 steps
LR decay: to 2.5% of peak
GPU: A100-SXM4-40GB (Colab)
Training time: ~2-3 hours
```

---

## Gripper Oscillation Diagnosis

Added action logging to the eval script to capture every action sent to the robot during inference. The `action_log.csv` (in `results/`) reveals the root cause.

### Gripper Action During Inference (60 seconds)

```
t=0.0s   gripper=0.48   (open)
t=5.6s   gripper=31.04  (closed)
t=11.8s  gripper=2.29   (open!)
t=18.1s  gripper=36.53  (closed)
t=24.6s  gripper=43.57  (closed)
t=31.3s  gripper=-0.11  (open!)
t=37.7s  gripper=32.78  (closed)
t=44.3s  gripper=34.84  (closed)
t=51.0s  gripper=17.36  (transitioning)
t=57.4s  gripper=37.24  (closed)
```

The gripper **oscillates wildly** between open (0) and closed (35-49). Max step-to-step change: **30.1 degrees** in a single step.

### Joint Jitter Analysis

| Joint | Mean Step-to-Step Change | Max Change |
|-------|--------------------------|------------|
| shoulder_pan | 0.76 | 5.13 |
| shoulder_lift | 1.63 | 16.17 |
| elbow_flex | 1.42 | 17.17 |
| wrist_flex | 1.64 | 13.27 |
| wrist_roll | 0.52 | 5.87 |
| **gripper** | **2.45** | **30.11** |

### Root Cause: Training Data Imbalance

The gripper action distribution in the training data is severely skewed:

| Range | Percentage | Description |
|-------|-----------|-------------|
| 0-5 | **76%** | Gripper open (approach phase) |
| 5-30 | 11% | Transition |
| 30-48 | **12%** | Gripper grasping |

With MEAN_STD normalization:
- **Open (0)** normalizes to **-0.54 sigma** — easy to predict
- **Grasp (35)** normalizes to **+2.21 sigma** — model must output a rare, high-confidence prediction

The model's loss is dominated by the 76% open-gripper frames. It learns approach behavior well but can't commit to the grasp — it oscillates between "keep approaching" and "grasp now."

### Conclusion

SmolVLA 500M lacks the capacity to learn precise grasp timing from this data distribution. The solution is to move to a larger model (Pi0.5, ~3B params) with LoRA fine-tuning.

---

## Problems Faced & Solutions

### Critical Issues (Solved)

| # | Issue | Impact | Solution |
|---|-------|--------|----------|
| 1 | **Language instruction mismatch** | 0% success | Use exact training task string |
| 2 | **Camera configuration swap** | Complete spatial failure | Verify camera feeds visually, match device paths |
| 3 | **Overfitting (15.87 epochs)** | Poor generalization | Reduced to 6.5 epochs |
| 4 | **Robot calibration drift** | Imprecise grasps | Recalibrate before every eval session |
| 5 | **Camera angle shift** | Visual distribution mismatch | Preview tool to compare with training data |
| 6 | **Starting state inconsistency** | Variable performance | Manual reset to home position between episodes |
| 7 | **Config compatibility** | `compile_model` DraccusError | Remove unsupported fields from Hub config |

### Diagnosed (Unsolved by SmolVLA)

| # | Issue | Root Cause | Next Step |
|---|-------|------------|-----------|
| 8 | **Gripper oscillation** | 500M model can't learn grasp timing from imbalanced data (76% open) | Move to Pi0.5 (~3B) with LoRA |

For detailed write-ups of each issue, see [docs/challenges_and_solutions.md](docs/challenges_and_solutions.md).

---

## Scripts & Notebooks

All scripts are in the `scripts/` directory:

### Training Notebooks

| File | Description |
|------|-------------|
| `smolvla_final_training.ipynb` | SmolVLA v9 training (HPO-optimized, cleaned dataset, Colab A100) |
| `smolvla_optuna_hpo.ipynb` | Optuna HPO — local (RTX 3050) |
| `smolvla_optuna_hpo_colab.ipynb` | Optuna HPO — Colab (30 trials) |
| `pi05_colab_so101_training.ipynb` | Pi0.5 LoRA training (Colab, planned) |

### Eval & Utility Scripts

| File | Description |
|------|-------------|
| `run_smolvla_eval.sh` | SmolVLA evaluation with RTC on SO-101 |
| `run_pi05_eval.sh` | Pi0.5 evaluation with RTC on SO-101 |
| `preview_cameras.py` | Live camera preview for angle verification |

### Diagnostic Data

| File | Description |
|------|-------------|
| `results/action_log.csv` | 474 steps of raw action predictions showing gripper oscillation |

**Note**: All API tokens have been replaced with placeholders (`YOUR_HF_TOKEN_HERE`, `YOUR_WANDB_KEY_HERE`).

---

## Tools & Technologies

| Category | Tools |
|----------|-------|
| **ML Framework** | PyTorch, HuggingFace Transformers |
| **Robot Learning** | LeRobot (HuggingFace) |
| **HPO** | Optuna (30 trials, pruning) |
| **Experiment Tracking** | Weights & Biases |
| **Model Hosting** | HuggingFace Hub |
| **Training Compute** | Google Colab (A100, L4) |
| **Local GPU** | NVIDIA RTX 3050 Laptop |
| **Robot Hardware** | SO-101 Leader + Follower arms, Feetech servos |
| **Cameras** | 2x USB (front + wrist), 640x480 @ 30fps |

---

## Installation

```bash
# Clone repository
git clone https://github.com/Adithya191101/hugging-face_So101.git
cd hugging-face_So101

# Install LeRobot
cd ~/lerobot
pip install -e ".[feetech]"   # SO-101 hardware
pip install -e ".[smolvla]"   # SmolVLA model

# Login
huggingface-cli login
wandb login  # optional
```

---

## HuggingFace Resources

- **Profile**: [huggingface.co/AdithyaRajendran](https://huggingface.co/AdithyaRajendran)
- **Dataset**: [so101_grab_brain_t2](https://huggingface.co/datasets/AdithyaRajendran/so101_grab_brain_t2) (239 episodes, 99,845 frames)
- **Latest SmolVLA**: [smolvla_so101_grab_brain_t2_v9](https://huggingface.co/AdithyaRajendran/smolvla_so101_grab_brain_t2_v9)
- **LeRobot**: [github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)

---

## Videos

### ACT Model - 80% Success Rate

<video src="https://github.com/Adithya191101/hugging-face_So101/raw/main/videos/ACT_policy.mp4" controls width="100%"></video>

### SmolVLA - Language-Conditioned Grasping

<video src="https://github.com/Adithya191101/hugging-face_So101/raw/main/videos/SmolVLA.mp4" controls width="100%"></video>

### Data Collection via Teleoperation

<video src="https://github.com/Adithya191101/hugging-face_So101/raw/main/videos/Imitation.mp4" controls width="100%"></video>

---

## Future Work

### Immediate
- [ ] Train Pi0.5 with LoRA on Colab (A100) using same dataset
- [ ] Evaluate Pi0.5 on SO-101

### Short-term
- [ ] Compare Pi0.5 vs SmolVLA grasping precision
- [ ] If Pi0.5 works, optimize RTC parameters for it
- [ ] Record grasp-focused demonstrations if needed

### Long-term
- [ ] Multi-task VLA (pick, place, push, stack)
- [ ] Diverse language instruction training
- [ ] Open-source contribution to LeRobot

---

## Contact

**Adithya Rajendran**
- GitHub: [@Adithya191101](https://github.com/Adithya191101)
- HuggingFace: [@AdithyaRajendran](https://huggingface.co/AdithyaRajendran)

---

Apache 2.0 License

*Last Updated: March 2026*
