# Demo Videos

Demonstration videos arranged chronologically, telling the full story from data collection to 85% success.

---

## Video Demonstrations

### 1. Data Collection via Teleoperation

https://github.com/user-attachments/assets/7b4ca3ef-335a-4bf6-befb-b0d2956e9aa5

**Description**: 239 episodes collected via SO-101 leader arm teleoperation. Dual cameras (front + wrist) at 640x480 @ 30fps.

**File**: Imitation.mp4 (6.3 MB)

---

### 2. ACT Model — 80% Success Rate (Baseline)

https://github.com/user-attachments/assets/b1ddd207-edee-48ed-bd7f-71b98955f406

**Description**: ACT (Action Chunking Transformer) [Zhao et al., 2023] performing pick-and-place on deformable brain toy. Deterministic trajectories with temporal ensembling. Established 80% baseline but brittle to camera/lighting changes.

**File**: ACT_policy.mp4 (19 MB)

---

### 3. SmolVLA Phase 1 — Language-Conditioned Grasping (~33%)

https://github.com/user-attachments/assets/3799a5ea-dcf2-4eb9-a4b8-d78f636db92d

**Description**: SmolVLA Phase 1 (v1-v9). Robot approaches correctly but gripper oscillates. Diagnosed as training methodology issue — led to curriculum training approach in Phase 2.

**File**: SmolVLA.mp4 (5.1 MB)

---

### 4. SmolVLA Phase 2 — 85% Success Rate (Robot Camera View)

**Description**: SmolVLA Proven v3 with curriculum training + inference optimization. Dual-camera side-by-side view from robot's perspective. Shows consistent grasp and place behavior with deformable object.

- Curriculum training: 3 iterations (v1→v2→v3) [Bengio et al., 2009]
- Inference: num_steps=20, n_action_steps=10
- Flow matching denoising [Lipman et al., 2023]

**File**: smolvla_eval_v3.mp4 (19 MB)

---

### 5. SmolVLA Phase 2 — Full Robot View (External Camera)

**Description**: External phone camera showing the complete SO-101 workspace during continuous 85% success evaluation. Demonstrates camera displacement robustness and lighting variation handling. Policy ran continuously without restart between episodes.

**File**: SmolVLA_v3_phone.mp4 (32 MB)

---

## Video Specifications

| Video | Description | Format | Size |
|-------|-------------|--------|------|
| Imitation.mp4 | Data collection via teleoperation | MP4, 640x480 | 6.3 MB |
| ACT_policy.mp4 | ACT baseline — 80% success | MP4, 640x480 | 19 MB |
| SmolVLA.mp4 | Phase 1 — ~33% success | MP4, 640x480 | 5.1 MB |
| smolvla_eval_v3.mp4 | Phase 2 — 85% robot camera | MP4, 1296x550 | 19 MB |
| SmolVLA_v3_phone.mp4 | Phase 2 — 85% external camera | MP4 | 32 MB |

---

## Related Links

- **Dataset**: [AdithyaRajendran/so101_grab_brain_t2](https://huggingface.co/datasets/AdithyaRajendran/so101_grab_brain_t2)
- **SmolVLA Proven v3**: [smolvla_so101_grab_brain_t2_proven_v3](https://huggingface.co/AdithyaRajendran/smolvla_so101_grab_brain_t2_proven_v3)
- **ACT Model**: [so101_act_policy_V2](https://huggingface.co/AdithyaRajendran/so101_act_policy_V2)
- **LeRobot**: [github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)

---

*Last Updated: March 2026*
