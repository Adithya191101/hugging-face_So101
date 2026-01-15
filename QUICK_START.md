# Quick Start Guide

Get started with this robot learning project in 3 steps:

## 1. Clone and Setup

```bash
# Clone this repository
git clone https://github.com/Adithya191101/hugging-face_So101.git
cd hugging-face_So101

# Install LeRobot
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .
```

## 2. Try Pre-trained Models

### ACT Model (80% Success Rate)
```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --policy.path=AdithyaRajendran/so101_act_policy \
  --dataset.repo_id=test_act_deployment
```

### SmolVLA Model (Language-Conditioned)
```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --policy.path=AdithyaRajendran/so101_smolvla_policy_FINAL_v5 \
  --dataset.single_task="Grab the brain" \
  --dataset.repo_id=test_smolvla_deployment
```

## 3. Explore Documentation

- **[README.md](README.md)** - Complete project overview
- **[docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md)** - All commands and configurations
- **[docs/PROJECT_WORKFLOW.md](docs/PROJECT_WORKFLOW.md)** - Step-by-step workflow
- **[docs/challenges_and_solutions.md](docs/challenges_and_solutions.md)** - Problems I faced

## Key Resources

- **Datasets:** https://huggingface.co/datasets/AdithyaRajendran
- **Models:** https://huggingface.co/AdithyaRajendran
- **LeRobot:** https://github.com/huggingface/lerobot

## Need Help?

See [docs/challenges_and_solutions.md](docs/challenges_and_solutions.md) for common issues and debugging tips.
