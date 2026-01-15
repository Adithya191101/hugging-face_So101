# Demo Videos

This folder contains demonstration videos of the ACT and SmolVLA models in action.

---

## 📹 Video Demonstrations

### 1. ACT Model - Pick-and-Place (80% Success Rate)

<video src="https://github.com/Adithya191101/hugging-face_So101/raw/main/videos/ACT_policy.mp4" controls width="100%"></video>

**Description**: ACT (Action Chunking Transformer) model performing pick-and-place tasks on soft irregular objects

**Key Highlights**:
- Smooth, coordinated motion through temporal action chunking
- Successful grasping of deformable brain toy
- Demonstrates 80% success rate achievement
- Shows temporal ensembling for fluid trajectories

**Performance**: 80% success rate on soft irregular round/obloid objects
**File**: ACT_policy.mp4 (19 MB)

---

### 2. SmolVLA - Language-Conditioned Grasping

<video src="https://github.com/Adithya191101/hugging-face_So101/raw/main/videos/SmolVLA.mp4" controls width="100%"></video>

**Description**: SmolVLA (Small Vision-Language-Action) model performing language-conditioned grasping

**Key Highlights**:
- Language-conditioned task execution: "Grab the brain"
- Vision-language model (500M parameters) for robotic manipulation
- Demonstrates current 33% success rate after debugging
- Shows object approach and grasping behavior

**Current Status**: 33% success rate, working towards >70% with generalization improvements
**Model**: SmolVLM2-500M-Video-Instruct fine-tuned for robotic manipulation
**File**: SmolVLA.mp4 (5.1 MB)

---

### 3. Data Collection via Teleoperation

<video src="https://github.com/Adithya191101/hugging-face_So101/raw/main/videos/Imitation.mp4" controls width="100%"></video>

**Description**: Imitation learning demonstration showing teleoperation and data collection process

**Key Highlights**:
- Shows the data collection methodology using SO-101 leader arm
- Demonstrates how human demonstrations are recorded
- Illustrates the teleoperation setup for training data
- Foundation for both ACT and SmolVLA training datasets

**Training Data**: 241 episodes, 100,832 frames collected via teleoperation
**File**: Imitation.mp4 (6.3 MB)

---

## 🎯 Video Usage

### For Resume/Portfolio
- **ACT_policy.mp4**: Best demonstration of successful implementation (80% success)
- **SmolVLA.mp4**: Shows cutting-edge VLA research and debugging process
- **Imitation.mp4**: Demonstrates data collection methodology

### For Interviews
- Use ACT video to show successful implementation
- Use SmolVLA video to discuss debugging challenges (language mismatch, camera swap)
- Use Imitation video to explain training data pipeline

### For GitHub Visitors
All videos demonstrate:
- Real robot hardware (SO-101 6-DOF arm)
- Dual-camera setup (front + wrist views)
- End-to-end pipeline from teleoperation to autonomous execution

---

## 📊 Technical Details

### Hardware
- **Robot**: SO-101 Follower Arm (6-DOF)
- **Teleoperation**: SO-101 Leader Arm
- **Cameras**:
  - camera1: Front view (scene/table overview)
  - camera2: Wrist view (gripper close-up)
- **Object**: Soft irregular brain toy (deformable)

### Software Stack
- **Framework**: LeRobot (HuggingFace)
- **Models**: ACT, SmolVLA
- **Training**: Google Colab (V100/A100 GPUs)
- **Tracking**: Weights & Biases

### Dataset
- **Episodes**: 241 successful demonstrations
- **Frames**: 100,832 total frames
- **FPS**: 30 frames per second
- **Task**: Language-conditioned grasping ("Grab the brain")

---

## 🔗 Related Links

- **Training Dataset**: [AdithyaRajendran/so101_grab_brain_t2](https://huggingface.co/datasets/AdithyaRajendran/so101_grab_brain_t2)
- **ACT Model**: [AdithyaRajendran/so101_act_policy](https://huggingface.co/AdithyaRajendran/so101_act_policy)
- **SmolVLA Model**: [AdithyaRajendran/so101_smolvla_policy_FINAL_v5](https://huggingface.co/AdithyaRajendran/so101_smolvla_policy_FINAL_v5)
- **LeRobot Framework**: [github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)

---

## 🎬 Video Specifications

| Video | Duration | Resolution | Format | Size |
|-------|----------|------------|--------|------|
| ACT_policy.mp4 | ~30-60s | 640x480 | MP4 | 19 MB |
| SmolVLA.mp4 | ~30-60s | 640x480 | MP4 | 5.1 MB |
| Imitation.mp4 | ~30-60s | 640x480 | MP4 | 6.3 MB |

---

## 📝 Notes

- Videos demonstrate real-world performance, not simulations
- All footage recorded on actual SO-101 hardware
- Videos show both successful and challenging scenarios
- Dual-camera views provide comprehensive perspective

---

**Last Updated**: January 15, 2026
**Status**: ✅ Videos Added
**Next**: Add W&B training curves and analysis plots to results/ folder
