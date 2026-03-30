"""
SmolVLA Remote Inference Server for RunPod.

Loads a SmolVLA policy and serves inference via HTTP.
The client (running on your local PC) sends camera images + robot state,
and receives back an action chunk (50 steps).

Usage on RunPod:
    pip install -e ".[smolvla]" && pip install fastapi uvicorn python-multipart
    python smolvla_server.py --policy_repo AdithyaRajendran/smolvla_so101_grab_brain_t2_v9

The server listens on port 8000. Expose this port on RunPod.
"""

import argparse
import base64
import io
import logging
import time

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SmolVLA Inference Server")

# Global state
policy = None
preprocessor = None
postprocessor = None
device = None


class ObservationRequest(BaseModel):
    """Observation sent from the client."""
    camera1_jpg_b64: str  # front camera, JPEG base64
    camera2_jpg_b64: str  # wrist camera, JPEG base64
    state: list[float]    # 6 joint positions
    task: str             # language task description


class ActionResponse(BaseModel):
    """Action chunk returned to the client."""
    actions: list[list[float]]  # 50 x 6 action chunk
    inference_time_ms: float


def decode_image(jpg_b64: str) -> np.ndarray:
    """Decode base64 JPEG to numpy array (H, W, 3) uint8."""
    from PIL import Image
    jpg_bytes = base64.b64decode(jpg_b64)
    img = Image.open(io.BytesIO(jpg_bytes)).convert("RGB")
    return np.array(img)


def build_observation_tensors(
    camera1: np.ndarray,
    camera2: np.ndarray,
    state: list[float],
    task: str,
) -> dict:
    """Build observation dict matching eval_with_real_robot.py exactly (lines 300-315)."""
    obs = {}

    # Match eval script: images go through same transform pipeline
    # (H, W, 3) uint8 -> float32/255 -> permute(2,0,1) -> unsqueeze(0) -> to(device)
    for name, img in [("observation.images.front", camera1), ("observation.images.wrist", camera2)]:
        t = torch.from_numpy(img).float() / 255.0
        t = t.permute(2, 0, 1).contiguous()
        t = t.unsqueeze(0).to(device)
        obs[name] = t

    # State: unsqueeze(0) + to(device) — matching eval script
    obs["observation.state"] = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    # Task
    obs["task"] = [task]
    obs["robot_type"] = "so_follower"

    return obs


@app.post("/predict", response_model=ActionResponse)
async def predict(req: ObservationRequest):
    """Run SmolVLA inference on an observation and return action chunk."""
    t_start = time.perf_counter()

    # Decode images
    cam1 = decode_image(req.camera1_jpg_b64)
    cam2 = decode_image(req.camera2_jpg_b64)

    # Build observation tensors
    obs = build_observation_tensors(cam1, cam2, req.state, req.task)

    # Preprocess
    preprocessed = preprocessor(obs)

    # Inference
    with torch.no_grad():
        actions = policy.predict_action_chunk(preprocessed)

    # Postprocess (unnormalize)
    actions = postprocessor(actions)
    actions = actions.squeeze(0).cpu()  # (50, action_dim)

    # Extract only the 6 SO-101 joints (model may output padded dims)
    n_joints = 6
    actions = actions[:, :n_joints].tolist()

    inference_ms = (time.perf_counter() - t_start) * 1000
    logger.info(f"Inference: {inference_ms:.1f}ms, chunk shape: {len(actions)}x{len(actions[0])}")

    return ActionResponse(actions=actions, inference_time_ms=inference_ms)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": policy is not None, "num_steps": policy.config.num_steps if policy else None, "n_action_steps": policy.config.n_action_steps if policy else None}


def load_policy(policy_repo: str, device_str: str, num_steps: int = None, n_action_steps: int = None):
    """Load SmolVLA policy and processors."""
    global policy, preprocessor, postprocessor, device

    device = torch.device(device_str)
    logger.info(f"Loading policy from {policy_repo} on {device}...")

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    # Load config
    config = PreTrainedConfig.from_pretrained(policy_repo)
    if num_steps is not None:
        config.num_steps = num_steps
    if n_action_steps is not None:
        config.n_action_steps = n_action_steps
    logger.info(f"Loading {policy_repo} on {device} with num_steps={config.num_steps}, n_action_steps={config.n_action_steps}...")
    policy_class = get_policy_class(config.type)

    # Load model
    if config.use_peft:
        from peft import PeftConfig, PeftModel
        peft_config = PeftConfig.from_pretrained(policy_repo)
        policy = policy_class.from_pretrained(peft_config.base_model_name_or_path, config=config)
        policy = PeftModel.from_pretrained(policy, policy_repo, config=peft_config)
    else:
        policy = policy_class.from_pretrained(policy_repo, config=config)

    policy = policy.to(device)
    policy.eval()
    logger.info("Policy loaded successfully")

    # Load preprocessor/postprocessor
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=policy_repo,
        dataset_stats=None,
        preprocessor_overrides={
            "device_processor": {"device": device_str},
        },
    )
    logger.info("Preprocessor/postprocessor loaded")

    # Warmup inference
    logger.info("Running warmup inference...")
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_obs = build_observation_tensors(dummy_img, dummy_img, [0.0] * 6, "warmup")
    dummy_pre = preprocessor(dummy_obs)
    with torch.no_grad():
        _ = policy.predict_action_chunk(dummy_pre)
    logger.info("Warmup complete, server ready!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmolVLA Inference Server")
    parser.add_argument("--policy_repo", type=str,
                        default="AdithyaRajendran/smolvla_so101_grab_brain_t2_v9")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_steps", type=int, default=None,
                        help="Override denoising steps (default: model's config, usually 10)")
    parser.add_argument("--n_action_steps", type=int, default=None,
                        help="Override action steps to execute per chunk (default: model's config)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    load_policy(args.policy_repo, args.device, args.num_steps, args.n_action_steps)
    uvicorn.run(app, host=args.host, port=args.port)
