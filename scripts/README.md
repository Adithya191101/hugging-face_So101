# Scripts

## Training Notebooks

| File | Description |
|------|-------------|
| `smolvla_final_training.ipynb` | SmolVLA curriculum training (Colab A100) |
| `smolvla_optuna_hpo.ipynb` | Optuna HPO — local (RTX 3050, 30 trials) |
| `smolvla_optuna_hpo_colab.ipynb` | Optuna HPO — Colab version |
| `pi05_colab_so101_training.ipynb` | Pi0.5 LoRA training (planned) |

## Remote Inference

| File | Description |
|------|-------------|
| `smolvla_server.py` | FastAPI inference server for RunPod GPU deployment. Serves SmolVLA model via HTTP. |
| `smolvla_client.py` | Multi-threaded robot client. Handles cameras, inference requests, action execution, and video recording. |

## Evaluation

| File | Description |
|------|-------------|
| `run_smolvla_eval.sh` | SmolVLA evaluation with RTC on SO-101 |
| `run_pi05_eval.sh` | Pi0.5 evaluation with RTC |
| `preview_cameras.py` | Live camera preview for angle verification |

## Diagnostic Data

| File | Description |
|------|-------------|
| `../results/action_log.csv` | 474 steps of raw action predictions (gripper oscillation analysis) |

*All API tokens have been replaced with placeholders.*
