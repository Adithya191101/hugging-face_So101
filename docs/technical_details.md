# Vision-Language-Action Models for Robotic Manipulation

**Project Duration**: December 2024 - January 2025
**Organization**: Independent Research
**Hardware**: SO-101 Follower Robotic Arm, Dual Camera Setup (Front + Wrist Mounted)

## 🎯 Project Overview

Implemented and evaluated state-of-the-art vision-language-action (VLA) models for robotic pick-and-place tasks on a low-cost SO-101 robotic manipulator. Focused on training generalist policies that can interpret natural language instructions and execute manipulation tasks in real-world environments.

**GitHub Repository**: [AdithyaRajendran/robot-vla-manipulation](https://github.com/AdithyaRajendran/robot-vla-manipulation) *(to be published)*

---

## 🤖 Models Implemented & Results

### 1. Action Chunking Transformer (ACT)

**Task**: Pick and place of soft, irregular round/obloid objects (brain toy)

**Achievements**:
- ✅ **80% success rate** on pick-and-place tasks with irregular soft objects
- Trained on 241 teleoperation demonstrations (~100K frames)
- Robust to object deformations and position variations
- Smooth, natural motion trajectories using dual-camera visual feedback

**Technical Details**:
- Policy: Action Chunking Transformer (ACT) with temporal ensembling
- Input: RGB images from front camera (640×480) + wrist camera (640×480) + joint states (6-DOF)
- Output: Action sequences (chunk_size=100, temporal ensembling)
- Training: 50K steps, batch_size=8, learning_rate=1e-5

**Key Learnings**:
- ACT's temporal action chunking provides smooth, coordinated motions
- Critical importance of diverse demonstration data quality
- Dual-camera setup significantly improves spatial reasoning

---

### 2. SmolVLA (Small Vision-Language-Action Model)

**Task**: Language-conditioned pick and place with natural language instructions

**Current Status**:
- 🔬 **In Progress**: Trained to operate in constrained pickup area (2×2 inch workspace)
- Working on generalizing to larger pickup areas and varied object positions
- Investigating vision-language conditioning for multi-task capabilities

**Achievements to Date**:
- Successfully trained 5 model iterations (v2-v6) exploring hyperparameter optimization
- Identified and resolved critical issues:
  - Language instruction matching between training and deployment
  - Camera configuration and visual observation consistency
  - Gripper action prediction and temporal smoothness
- Developed robust training pipeline with automatic dataset versioning

**Technical Details**:
- Base Model: SmolVLM2-500M-Video-Instruct (HuggingFace)
- Architecture: Frozen vision encoder + trainable action decoder
- Training Dataset: 241 episodes (100K frames) + deployment mix augmentation (in progress)
- Hyperparameters:
  - chunk_size: 30 (1-second prediction horizon)
  - n_action_steps: 20 (balanced smoothness and reactivity)
  - batch_size: 32
  - learning_rate: 1e-5
  - Training steps: 20K-25K (6-7 epochs)
  - Image augmentation: ColorJitter, RandomAffine, SharpnessJitter

**Experimental Findings**:
1. **Language Conditioning Critical**:
   - Task description must match exactly between training and deployment
   - Model showed 0% success with mismatched instructions, improved with correct matching

2. **Visual Observation Consistency**:
   - Camera configuration mismatch caused complete policy failure
   - Swapped camera feeds led to spatial reasoning errors (robot targeting wrong objects)

3. **Action Smoothness vs Reactivity Trade-off**:
   - chunk_size=10: More reactive but jerky motion (10-26° discontinuities)
   - chunk_size=30: Smoother motion but less reactive to visual feedback
   - Optimal: chunk_size=30 with n_action_steps=20

4. **Overfitting Diagnosis**:
   - 15.87 epochs → severe overfitting on 241-episode dataset
   - 6.35 epochs → optimal generalization
   - Gripper action distribution: Training (54.5% closed) vs Failed Deployment (0% closed)

**Next Steps**:
- Complete deployment mix dataset (50 episodes in actual environment)
- Train v6 model on merged dataset (241 + 50 = 291 episodes)
- Target: >70% success rate on generalized pickup area
- Expand language instruction diversity for multi-task capabilities

---

## 🛠️ Technical Stack & Tools

### Machine Learning & Training
- **LeRobot Framework**: End-to-end robot learning pipeline (data collection, training, deployment)
- **HuggingFace Hub**: Model hosting, dataset versioning, collaborative ML
  - Published datasets: `AdithyaRajendran/so101_grab_brain_t2` (100K frames)
  - Model checkpoints: Multiple versions with full reproducibility
- **Weights & Biases (W&B)**: Experiment tracking, hyperparameter optimization
  - Tracked 5+ training runs with loss curves, gradient norms, action distributions
  - Real-time monitoring of training stability and overfitting

### Development & Deployment
- **Google Colab**: GPU-accelerated training (V100, A100)
  - Utilized Google Drive for checkpoint persistence
  - Automatic Mixed Precision (AMP) for memory efficiency
- **PyTorch**: Deep learning framework with custom policy implementations
- **Python Libraries**:
  - `datasets` (HuggingFace): Efficient large-scale dataset handling
  - `transformers`: Pre-trained vision-language models
  - `opencv-python`: Camera interface and image processing
  - `pandas`, `numpy`: Data analysis and manipulation

### Hardware & Robotics
- **SO-101 Follower Arm**: 6-DOF robotic manipulator
- **Feetech Servos**: Motor control with position feedback
- **Dual Camera Setup**:
  - Front camera: Scene understanding (640×480 @ 30fps)
  - Wrist camera: Fine-grained manipulation (640×480 @ 30fps)
- **SO-101 Leader Arm**: Teleoperation for demonstration collection

### Version Control & Documentation
- **Git/GitHub**: Code versioning, documentation, collaboration
- **Markdown**: Technical documentation and experiment logs

---

## 📊 Key Metrics & Performance

### ACT Model
| Metric | Value |
|--------|-------|
| Success Rate (Pick & Place) | 80% |
| Training Episodes | 241 |
| Training Frames | ~100,000 |
| Object Type | Soft irregular (brain toy) |
| Inference Speed | 30 FPS |

### SmolVLA Model (v5 - Latest Stable)
| Metric | Value |
|--------|-------|
| Training Dataset Size | 241 episodes (100,832 frames) |
| Model Parameters | ~500M (frozen vision) + ~50M (trainable action head) |
| Training Time | 3.5 hours (V100 GPU) |
| Training Steps | 20,000 |
| Epochs | 6.35 |
| Final Loss | 0.012 |
| Batch Size | 32 |
| Current Success Rate | In evaluation (deployment mix in progress) |
| Working Pickup Area | 2×2 inch (constrained) |
| Target Pickup Area | 6×6 inch (generalizable) |

### Data Collection
| Metric | Value |
|--------|-------|
| Total Demonstrations | 241+ episodes |
| Total Frames Collected | 100,000+ |
| Teleoperation FPS | 30 |
| Average Episode Duration | 20-25 seconds |
| Camera Resolution | 640×480 RGB per camera |

---

## 🎓 Key Learnings & Contributions

### Technical Insights

1. **Vision-Language Alignment**:
   - Discovered critical importance of exact language instruction matching
   - Model performance degraded from partial success to 0% with mismatched instructions
   - Learned that VLA models create distinct embeddings for similar but non-identical phrases

2. **Visual Observation Consistency**:
   - Camera configuration must be EXACTLY reproducible between training and deployment
   - Even camera feed swaps cause catastrophic spatial reasoning failures
   - Implemented systematic camera verification protocols

3. **Hyperparameter Optimization for Small Datasets**:
   - Developed epoch-based training strategy (6-7 epochs optimal for ~250 episodes)
   - Batch size 32 optimal for efficiency; overfitting controlled by epoch count, not batch size
   - Identified chunk_size/n_action_steps trade-off for smooth yet reactive policies

4. **Overfitting Diagnosis in Robot Learning**:
   - Created diagnostic pipeline analyzing gripper action distributions
   - Training: 54.5% closed frames → Overfitted deployment: 0% closed frames
   - Developed mixed-dataset training strategy to improve generalization

5. **Multi-Environment Training Strategy**:
   - Proposed and implementing mixed-dataset approach
   - Combining original training data + deployment environment demonstrations
   - Expected to improve generalization from 2×2 inch to 6×6 inch pickup area

### Software Engineering Practices

- **Reproducible Research**: All experiments tracked with W&B, datasets versioned on HuggingFace Hub
- **Systematic Debugging**: Root cause analysis of failures (language mismatch, camera swap, overfitting)
- **Iterative Development**: 5 model versions with progressive improvements
- **Data-Centric Approach**: Focus on data quality and environment consistency

---

## 📁 Repository Structure *(To Be Published)*

```
robot-vla-manipulation/
├── README.md                          # Project overview and setup
├── docs/
│   ├── act_training.md               # ACT model documentation
│   ├── smolvla_training.md           # SmolVLA training guide
│   └── troubleshooting.md            # Common issues and solutions
├── configs/
│   ├── act_config.yaml               # ACT hyperparameters
│   └── smolvla_config.yaml           # SmolVLA hyperparameters
├── scripts/
│   ├── record_demonstrations.sh      # Data collection
│   ├── train_act.sh                  # ACT training
│   ├── train_smolvla.sh              # SmolVLA training
│   ├── merge_datasets.py             # Dataset utilities
│   └── evaluate_policy.sh            # Deployment testing
├── analysis/
│   ├── analyze_gripper_actions.py    # Action distribution analysis
│   ├── compare_training_eval.py      # Training vs deployment analysis
│   └── visualize_trajectories.py    # Trajectory visualization
├── videos/
│   ├── act_successful_grasps/        # ACT demo videos
│   ├── smolvla_training/             # SmolVLA training progress
│   └── failure_analysis/             # Diagnostic videos
└── results/
    ├── wandb_logs/                   # Training metrics
    └── evaluation_results/           # Deployment statistics
```

---

## 🎥 Videos & Demonstrations *(To Be Added)*

### ACT Model
- [ ] Successful pick-and-place demonstrations (10 episodes)
- [ ] Failure case analysis
- [ ] Training progress visualization

### SmolVLA Model
- [ ] Training data collection (teleoperation)
- [ ] v5 deployment attempts (language mismatch issue)
- [ ] Camera configuration debugging
- [ ] v6 evaluation (deployment mix) - *In Progress*

---

## 📚 References & Resources

### Papers
1. **ACT**: "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (Zhao et al., 2023)
2. **SmolVLM**: "SmolVLM - Compact Vision-Language Model" (HuggingFace, 2024)
3. **LeRobot**: "LeRobot: Making AI-powered robotics more accessible" (HuggingFace, 2024)

### Links
- **LeRobot Framework**: https://github.com/huggingface/lerobot
- **HuggingFace Hub Profile**: https://huggingface.co/AdithyaRajendran
- **Training Dataset**: https://huggingface.co/datasets/AdithyaRajendran/so101_grab_brain_t2
- **Model Checkpoints**:
  - ACT: `AdithyaRajendran/so101_act_policy`
  - SmolVLA v5: `AdithyaRajendran/so101_smolvla_policy_FINAL_v5`
  - SmolVLA v6: `AdithyaRajendran/so101_smolvla_policy_FINAL_v6` *(Training in progress)*

---

## 🚀 Future Work

### Short-term (1-2 weeks)
- [ ] Complete deployment mix dataset collection (50 episodes)
- [ ] Train and evaluate SmolVLA v6 on merged dataset
- [ ] Achieve >70% success rate on generalized pickup area
- [ ] Publish GitHub repository with full documentation

### Medium-term (1-2 months)
- [ ] Expand to multi-object scenarios
- [ ] Implement diverse language instruction training
- [ ] Generalize to 6×6 inch pickup area
- [ ] Compare ACT vs SmolVLA performance head-to-head

### Long-term (3-6 months)
- [ ] Multi-task VLA training (pick, place, push, stack)
- [ ] Real-world robustness testing (varied lighting, backgrounds)
- [ ] Open-source contribution to LeRobot framework
- [ ] Publication-quality experimental results

---

## 💡 Skills Demonstrated

### Technical Skills
- Deep Learning (PyTorch, Transformers)
- Robot Learning (Imitation Learning, Vision-Language Models)
- Computer Vision (Multi-camera systems, Image preprocessing)
- MLOps (Experiment tracking, Model versioning, Reproducibility)
- Data Engineering (Large-scale dataset handling, Augmentation pipelines)

### Research Skills
- Systematic debugging and root cause analysis
- Hypothesis-driven experimentation
- Quantitative performance evaluation
- Technical documentation and communication

### Tools & Platforms
- Python, PyTorch, HuggingFace Ecosystem
- Google Colab, Weights & Biases
- Git, GitHub, Markdown
- LeRobot, OpenCV, NumPy, Pandas

---

## 📧 Contact

**Name**: Adithya Rajendran
**GitHub**: [AdithyaRajendran](https://github.com/AdithyaRajendran)
**HuggingFace**: [AdithyaRajendran](https://huggingface.co/AdithyaRajendran)
**Email**: [Your Email]

---

## 📄 License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **HuggingFace** for the LeRobot framework and model hosting infrastructure
- **SmolVLM Team** for the pre-trained vision-language model
- **Open-source robotics community** for inspiration and tools

---

*Last Updated: January 2026*
