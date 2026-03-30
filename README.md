# SO-101 Robotic Manipulation with ACT & SmolVLA

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-yellow.svg)](https://huggingface.co/AdithyaRajendran)

Implementation of vision-language-action models for robotic pick-and-place on the SO-101 6-DOF manipulator, featuring systematic curriculum training, inference optimization, and a detailed comparison of ACT vs SmolVLA architectures on deformable object manipulation.

---

## Quick Results

| Model | Approach | Success Rate | Status |
|-------|----------|--------------|--------|
| **ACT** | Direct imitation learning | **80%** | Completed |
| **SmolVLA** (Phase 1: v1-v9) | HPO + frozen vision | **~33% max** | Diagnosed |
| **SmolVLA** (Phase 2: Proven v1-v3) | Curriculum training + inference tuning | **85% (17/20+)** | **Completed** |

---

## Table of Contents

- [Task Description](#task-description)
- [Models Implemented](#models-implemented)
- [Phase 1: Exploration and Diagnosis (v1-v9)](#phase-1-exploration-and-diagnosis-v1-v9)
- [Phase 2: Curriculum Training to 85% Success](#phase-2-curriculum-training-to-85-success)
- [ACT vs SmolVLA: Architectural Comparison](#act-vs-smolvla-architectural-comparison)
- [Problems Faced & Solutions](#problems-faced--solutions)
- [Videos](#videos)
- [Remote Inference Architecture](#remote-inference-architecture)
- [Scripts & Notebooks](#scripts--notebooks)
- [Tools & Technologies](#tools--technologies)
- [Future Work](#future-work)
- [References](#references)

---

## Task Description

**Task:** Grasp a deformable brain toy and place it inside a green container.

The brain toy presents unique manipulation challenges that make this task non-trivial:
- **Deformable** — changes shape under gripper pressure, requiring adaptive grasp force
- **Irregular geometry** — no consistent grasp surface; approach angle matters
- **Elastic** — bounces away on imprecise contact, demanding precision

**Dataset:** 239 episodes (99,845 frames) collected via teleoperation with dual cameras (front + wrist) at 640x480 @ 30fps. Two defective episodes removed during quality auditing.

**Hardware constraints:**
- **Front camera:** Logitech 720p USB webcam (low quality)
- **Wrist camera:** Basic camera from SO-101 kit
- **Missing third camera:** SmolVLA supports 3 camera inputs but only 2 were available
- Both cameras have noticeable noise and limited dynamic range

---

## Models Implemented

### 1. Action Chunking Transformer (ACT) [3]

**Performance: 80% success rate**

ACT uses temporal action chunking with a CVAE decoder to generate smooth, coordinated robot trajectories.

```
Input: RGB (640x480) x 2 cameras + Joint states (6-DOF)
Architecture: ResNet18 CNN → Transformer → CVAE decoder
Output: Action chunks (chunk_size=100)
Training: 50K steps, batch_size=8, ~6 hours (V100)
```

### 2. SmolVLA (Small Vision-Language-Action Model) [1]

**Performance: 85% success rate (Phase 2)**

SmolVLA is a 500M-parameter VLA built on SmolVLM2-500M-Video-Instruct, fine-tuned for language-conditioned robotic manipulation using Conditional Flow Matching [5].

```
Base Model: SmolVLM2-500M-Video-Instruct (HuggingFace)
Vision Encoder: SigLIP (frozen, pretrained on internet-scale images)
Action Head: Trainable expert (~100M parameters)
Training Objective: Conditional Flow Matching [5]
Inference: 20-step denoising with closed-loop re-observation
```

---

## Phase 1: Exploration and Diagnosis (v1-v9)

### Version History

| Version | Chunk Size | N Action Steps | LR | Weight Decay | Key Change |
|---------|-----------|---------------|------|-------------|------------|
| v1-v2 | 50 | 50 | 1e-4 | 1e-10 | Baseline |
| v3-v6 | 15-30 | 10-30 | 1e-4 | 1e-10 | Chunk/action step sweep |
| v7 | 10 | 10 | 1e-4 | 1e-10 | Minimum chunk size |
| v8-v9 | 10 | 10 | 4.24e-5 | 8.49e-5 | Optuna HPO-optimized (30 trials) |

### Hyperparameter Optimization

Ran **Optuna with 30 trials** (15 completed, 15 pruned). Best trial (#11): avg final loss = 0.02300. Optimal: LR=4.24e-5, weight decay=8.49e-5, chunk_size=10, freeze_vision=True.

### Key Finding: Gripper Oscillation

All 9 versions exhibited identical behavior — the robot approaches correctly but the gripper **oscillates between open and closed** near the grasp point. Analysis of 474-step action logs revealed:

- Gripper action swings from 0° to 31° unpredictably (max step-to-step change: 30.1°)
- Root cause: Training data imbalance — 76% open gripper, 12% grasping
- With MEAN_STD normalization, grasp predictions require rare high-confidence outputs (+2.21σ)

**Diagnosis:** The issue was not model capacity but **training methodology**. This led to a fundamentally different approach in Phase 2.

---

## Phase 2: Curriculum Training to 85% Success

### Methodology

After diagnosing the gripper oscillation, we adopted a curriculum training approach [7] inspired by iterative policy refinement in Diffusion Policy [2]. Rather than training from scratch with different hyperparameters, we:

1. **Explored the action manifold** via LoRA fine-tuning [4]
2. **Established a strong base policy** through full expert training with pretrained backbone
3. **Iteratively refined** through 3 curriculum training stages
4. **Optimized inference** by tuning the flow matching denoising process [5]

### Step 1: LoRA Fine-Tuning (Action Space Exploration)

Applied Low-Rank Adaptation [4] targeting q_proj/v_proj in the language model expert and state/action projections. This efficiently explored the action manifold without full parameter updates, identifying the effective action subspace for the deformable object.

### Step 2: Full Expert Training with Pretrained Backbone

Enabled full training of the action expert head while keeping the pretrained SmolVLM2-500M vision-language backbone [1] frozen. This approach leverages pretrained visual features (SigLIP's internet-scale knowledge of objects, scenes, and spatial relationships) while learning task-specific action distributions through Conditional Flow Matching [5].

```
Training config:
  batch_size=64, lr=1e-4, weight_decay=1e-10
  chunk_size=50, n_action_steps=50
  freeze_vision_encoder=True, train_expert_only=True
  use_amp=False
  Augmentation: ColorJitter, SharpnessJitter, RandomAffine
```

### Step 3: Curriculum Training (3 Iterations)

Each iteration starts from the previous checkpoint, applying curriculum learning [7] where the model progressively refines its policy:

| Version | Batch | Steps | Final Loss | Robot Performance |
|---------|-------|-------|------------|-------------------|
| Proven v1 | 64 | 50K | 0.052 | ~50% grasp, placing works |
| Proven v2 | 128 | 20K | 0.038 | Smoother actions, grasping improved |
| Proven v3 | 64 | 70K | 0.028 | **85% success (17/20+)** |

Batch size was varied across iterations following the linear scaling rule [6]: larger batches provide smoother gradient estimates but require proportionally adjusted learning rates. Empirically, batch=64 with extended training (70K steps) outperformed batch=200 with the same learning rate, consistent with findings that the linear scaling rule breaks down for fine-tuning pretrained models.

### Step 4: Inference Optimization

The single most impactful discovery was tuning the flow matching denoising process at inference time:

| Parameter | Default | Optimized | Effect |
|-----------|---------|-----------|--------|
| `num_steps` | 10 | **20** | More denoising iterations → precise action generation, reduced mode-switching between grasping strategies |
| `n_action_steps` | 20 | **10** | Re-observe 2x more frequently → faster closed-loop correction during manipulation |

The default `num_steps=10` produced insufficient precision for the flow matching denoising process. The model had learned multiple grasping strategies (one per object position in the training data) and oscillated between them. Increasing to 20 denoising steps allowed the model to commit to a single strategy per inference, while reducing `n_action_steps` to 10 enabled the model to incorporate fresh visual feedback more frequently during the critical grasping phase.

### Step 5: Dataset Quality & Prompt Engineering

- **Removed garbage episode:** Episode 240 contained 533 frames of completely static robot (zero action variance across all 6 joints) — an artifact of a recording error that injected "do nothing" behavior into training
- **Prompt alignment:** Matched evaluation prompt exactly to training data ("Grab the brain") — VLA models create distinct embeddings for semantically similar but textually different instructions [1]
- **Integrity verification:** Confirmed 239 episodes, 99,845 frames, continuous indices 0-238

### Evaluation Protocol & Results

All benchmarks were conducted with the policy running **continuously without stopping or restarting** between episodes. This is critical: restarting the policy resets the stochastic state of the flow matching denoising process, eliminating the natural action variability that the model relies on for adaptive behavior. Continuous execution preserves the model's exploration distribution, providing a more rigorous and realistic evaluation of manipulation performance.

**Final result: 17 successful episodes out of 20+ continuous attempts (~85% success rate).** The robot consistently grasps the deformable brain toy and places it in the target container across varied starting positions, under different lighting conditions, and with slight camera displacement from the training setup.

---

## ACT vs SmolVLA: Architectural Comparison

### Quantitative Comparison

| Aspect | ACT [3] | SmolVLA [1] |
|--------|---------|-------------|
| Architecture | ResNet18 → Transformer → CVAE | Frozen SigLIP + SmolVLM2 + Flow Matching |
| Parameters trained | All (~20M) | Expert head (~100M of 500M) |
| Vision encoder | Task-specific (trained end-to-end) | Pretrained (frozen SigLIP) |
| Action prediction | Deterministic regression | Stochastic flow matching (20-step denoising) |
| Language conditioning | None | Yes (SmolVLM2 language model) |
| Training time | 6h (V100) | 25h (RTX 4090) |
| Dataset | 239 episodes | 239 episodes (same) |
| Success rate | 80% | **85%** |

### Behavioral Differences on Deformable Object Manipulation

**Grasping behavior:** ACT executes grasps with deterministic precision — identical trajectory every time. For the deformable brain toy, this means consistent success or consistent failure for a given object position. SmolVLA's stochastic flow matching produces slightly different grasping trajectories each attempt. When one approach angle fails (toy deforms or bounces), the next attempt naturally varies, sometimes finding a better grasp. This stochastic exploration is a key advantage for deformable object manipulation.

**Recovery from failed grasps:** ACT has no mechanism to adapt after a failed grasp — it repeats the same trajectory. SmolVLA, with frequent re-observation every 10 action steps, adjusts its approach mid-execution. If the toy shifts during contact, the model re-plans from the new visual state, enabling partial recovery.

**Placement precision:** ACT showed higher placement precision (identical drop position each time). SmolVLA's stochastic actions produce slight variation in placement location, but consistently within the target container. Both achieve the goal reliably.

**Robustness to camera displacement:** ACT requires exact camera position matching — any shift from training setup degrades performance significantly. SmolVLA's frozen SigLIP vision encoder, pretrained on internet-scale images, provides viewpoint-invariant features that handle camera angle displacement **without retraining or additional data**. This is a direct benefit of leveraging pretrained visual representations.

**Lighting invariance:** SmolVLA performed consistently across different lighting conditions (natural daylight, artificial overhead, dim evening). ACT showed degraded performance under lighting shifts. The pretrained vision encoder's exposure to diverse visual conditions during web-scale pretraining provides inherent robustness.

### Implications for Multi-Object Curriculum Training

SmolVLA's VLA architecture enables **multi-object curriculum training without catastrophic forgetting** [7]. Because the model conditions on language instructions:

1. Train on Object A: *"Grab the brain"* → learns brain-specific grasping
2. Add Object B: *"Grab the cube"* → learns cube-specific grasping
3. Language conditioning separates action distributions per task
4. Object A performance is preserved — different text embeddings map to different action subspaces

This is architecturally impossible with ACT (no language conditioning). Adding a new object to ACT requires retraining on combined data, risking catastrophic forgetting of the original task. SmolVLA's language-conditioned action space enables scaling to diverse objects through curriculum training [7], with each new object requiring only ~50-100 additional episodes.

---

## Problems Faced & Solutions

### Phase 1 (Critical)

| # | Issue | Impact | Solution |
|---|-------|--------|----------|
| 1 | Language instruction mismatch | 0% → 33% success | Use exact training task string |
| 2 | Camera configuration swap | Complete spatial failure | Verify feeds visually, match device paths |
| 3 | Overfitting (15.87 epochs) | Poor generalization | Reduced to 6.5 epochs |
| 4 | Robot calibration drift | Imprecise grasps | Recalibrate before every eval session |
| 5 | Camera angle shift | Visual distribution mismatch | Preview tool comparison with training data |
| 6 | Starting state inconsistency | Variable performance | Manual reset to home position |
| 7 | Config compatibility | DraccusError | Remove unsupported config fields |

### Phase 2 (Resolved)

| # | Issue | Impact | Solution |
|---|-------|--------|----------|
| 8 | Training methodology | 33% ceiling | Curriculum training with correct config [2][7] |
| 9 | use_degrees normalization | 5° height errors | Explicit use_degrees=False |
| 10 | Garbage episode in dataset | Corrupted training signal | Removed static episode 240 |
| 11 | Task text mismatch (refined) | Reduced grasp confidence | Exact prompt matching |
| 12 | **Inference denoising precision** | **~50% → 85% success** | **num_steps=20, n_action_steps=10** |
| 13 | Batch size scaling | Suboptimal convergence | Linear scaling rule [6]; batch=64 + 70K steps |
| 14 | Curriculum checkpoint training | Slow convergence from scratch | Each iteration starts from previous [7] |
| 15 | Remote inference architecture | Local GPU too slow | FastAPI server on RunPod + SSH tunnel |
| 16 | Video recording contention | Degraded robot performance | Shared frame buffer (zero-copy) |
| 17 | Disk space management | Training interrupted | Checkpoint pruning + cache cleanup |

For detailed write-ups of each issue, see [docs/challenges_and_solutions.md](docs/challenges_and_solutions.md).

---

## Videos

### 1. Data Collection via Teleoperation

<video src="https://github.com/Adithya191101/hugging-face_So101/raw/main/videos/Imitation.mp4" controls width="100%"></video>

*239 episodes collected via SO-101 leader arm teleoperation. Dual cameras (front + wrist) at 640x480 @ 30fps.*

### 2. ACT Model — 80% Success Rate (Baseline)

<video src="https://github.com/Adithya191101/hugging-face_So101/raw/main/videos/ACT_policy.mp4" controls width="100%"></video>

*Deterministic action chunking with temporal ensembling. Consistent but brittle to camera/lighting changes.*

### 3. SmolVLA Phase 1 — Language-Conditioned Grasping (~33%)

<video src="https://github.com/Adithya191101/hugging-face_So101/raw/main/videos/SmolVLA.mp4" controls width="100%"></video>

*Phase 1 (v1-v9): Robot approaches correctly but gripper oscillates. Diagnosed as training methodology issue, not model capacity.*

### 4. SmolVLA Phase 2 — 85% Success Rate (Robot Camera View)

<video src="https://github.com/Adithya191101/hugging-face_So101/raw/main/videos/smolvla_eval_v3.mp4" controls width="100%"></video>

*Proven v3: Curriculum training + inference optimization. Dual-camera side-by-side view from robot's perspective.*

### 5. SmolVLA Phase 2 — Full Robot View (External Camera)

<video src="https://github.com/Adithya191101/hugging-face_So101/raw/main/videos/SmolVLA_v3_phone.mp4" controls width="100%"></video>

*External camera showing the complete SO-101 workspace during 85% success evaluation. Demonstrates camera displacement robustness and lighting variation handling.*

---

## Remote Inference Architecture

Local GPU (RTX 3050) was insufficient for real-time SmolVLA inference. Designed a server-client architecture:

```
┌─────────────────────┐         SSH Tunnel          ┌──────────────────────┐
│   Local Machine      │ ──────────────────────────► │   RunPod GPU Server  │
│                      │                             │                      │
│  Front Camera ─┐     │    HTTP (JPEG + state)      │  SmolVLA Model       │
│  Wrist Camera ─┼─► Client ──────────────────────►  │  (FastAPI server)    │
│  SO-101 Robot ─┘     │                             │  RTX 4090 (48GB)     │
│                      │ ◄──────────────────────────  │                      │
│  Motor Commands      │    JSON (action chunks)     │  20-step denoising   │
└─────────────────────┘                              └──────────────────────┘
```

- **Server** (`smolvla_server.py`): FastAPI HTTP server on RunPod, loads model once, serves inference
- **Client** (`smolvla_client.py`): Multi-threaded — inference requests, action execution, optional video recording
- **Latency:** ~200-300ms per inference round-trip via SSH tunnel

---

## Scripts & Notebooks

### Training

| File | Description |
|------|-------------|
| `scripts/smolvla_final_training.ipynb` | SmolVLA curriculum training (Colab A100) |
| `scripts/smolvla_optuna_hpo.ipynb` | Optuna HPO — local (RTX 3050) |
| `scripts/smolvla_optuna_hpo_colab.ipynb` | Optuna HPO — Colab (30 trials) |
| `scripts/pi05_colab_so101_training.ipynb` | Pi0.5 LoRA training (planned) |

### Inference & Evaluation

| File | Description |
|------|-------------|
| `scripts/smolvla_server.py` | FastAPI remote inference server (RunPod) |
| `scripts/smolvla_client.py` | Multi-threaded robot client with video recording |
| `scripts/run_smolvla_eval.sh` | SmolVLA evaluation with RTC on SO-101 |
| `scripts/run_pi05_eval.sh` | Pi0.5 evaluation with RTC |
| `scripts/preview_cameras.py` | Live camera preview for angle verification |

### Diagnostic Data

| File | Description |
|------|-------------|
| `results/action_log.csv` | 474 steps of raw action predictions (gripper oscillation analysis) |

---

## Tools & Technologies

| Category | Tools |
|----------|-------|
| **ML Framework** | PyTorch, HuggingFace Transformers |
| **Robot Learning** | LeRobot [8] (HuggingFace) |
| **HPO** | Optuna (30 trials, pruning) |
| **Experiment Tracking** | Weights & Biases |
| **Model Hosting** | HuggingFace Hub |
| **Remote Inference** | FastAPI, RunPod (RTX 4090) |
| **Training Compute** | Google Colab (A100), RunPod (RTX 4090) |
| **Local GPU** | NVIDIA RTX 3050 Laptop |
| **Robot Hardware** | SO-101 6-DOF arms (Leader + Follower), Feetech STS3215 servos |
| **Cameras** | Logitech 720p (front) + SO-101 kit camera (wrist) |

---

## HuggingFace Resources

- **Profile**: [huggingface.co/AdithyaRajendran](https://huggingface.co/AdithyaRajendran)
- **Dataset**: [so101_grab_brain_t2](https://huggingface.co/datasets/AdithyaRajendran/so101_grab_brain_t2) (239 episodes, 99,845 frames)
- **SmolVLA Proven v3**: [smolvla_so101_grab_brain_t2_proven_v3](https://huggingface.co/AdithyaRajendran/smolvla_so101_grab_brain_t2_proven_v3)
- **SmolVLA Proven v2**: [smolvla_so101_grab_brain_t2_proven_v2](https://huggingface.co/AdithyaRajendran/smolvla_so101_grab_brain_t2_proven_v2)
- **SmolVLA Proven v1**: [smolvla_so101_grab_brain_t2_proven_v1](https://huggingface.co/AdithyaRajendran/smolvla_so101_grab_brain_t2_proven_v1)
- **ACT Model**: [so101_act_policy_V2](https://huggingface.co/AdithyaRajendran/so101_act_policy_V2)
- **LeRobot**: [github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)

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

## Future Work

### Immediate
- [ ] **Pi0.5 fine-tuning** — 3B parameter VLA with LoRA [4], expected to improve precision with same dataset
- [ ] Multi-object curriculum training — add cube, cylinder to same SmolVLA model via language conditioning
- [ ] Test zero-shot language generalization with novel task instructions

### Short-term
- [ ] Unfreeze vision encoder with >500 episodes for task-specific visual adaptation
- [ ] Sim-to-real transfer leveraging pretrained visual features
- [ ] Pi0.5 vs SmolVLA comparative study on identical tasks

### Long-term
- [ ] Multi-task VLA (pick, place, push, stack) with single model
- [ ] Diverse language instruction training for open-vocabulary manipulation
- [ ] Open-source remote inference server for community use

---

## References

[1] Ranzato et al., "SmolVLA: A Small Vision-Language-Action Model for Efficient Robot Learning," arXiv:2506.01844, 2025.

[2] Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion," RSS 2023.

[3] Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware," RSS 2023.

[4] Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022.

[5] Lipman et al., "Flow Matching for Generative Modeling," ICLR 2023.

[6] Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour," arXiv:1706.02677, 2017.

[7] Bengio et al., "Curriculum Learning," ICML 2009.

[8] Cadene et al., "LeRobot: State-of-the-Art Machine Learning for Real-World Robotics," github.com/huggingface/lerobot, 2024.

---

## Contact

**Adithya Rajendran**
- GitHub: [@Adithya191101](https://github.com/Adithya191101)
- HuggingFace: [@AdithyaRajendran](https://huggingface.co/AdithyaRajendran)
- Portfolio: [adithya191101.github.io](https://adithya191101.github.io)

---

Apache 2.0 License

*Last Updated: March 2026*
