# self_play_GPU.py

import numpy as np
import torch
from sb3_contrib.common.wrappers.action_masker import ActionMasker
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from callback_logging import CSVLoggerCallback
import os

from Chess_Environment import ChessEnv    # :contentReference[oaicite:3]{index=3}
from action_mapping    import move_to_index  # :contentReference[oaicite:4]{index=4}

# 1) Build a correct mask_fn that works on ChessEnv directly
def mask_fn(env) -> np.ndarray:
    """
    env here *is* your ChessEnv (or a thin wrapper),
    so just use its board to build a legal‐move mask.
    """
    mask = np.zeros(env.action_space.n, dtype=bool)
    for mv in env.board.legal_moves:
        mask[ move_to_index[mv.uci()] ] = True
    return mask

def main():
    # ensure models folder exists
    os.makedirs("models", exist_ok=True)

    # 2) Choose device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 3) Instantiate and wrap your env exactly as in training
    base_env = ChessEnv()
    env      = ActionMasker(base_env, mask_fn)

    # 4) Build your MaskablePPO
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        verbose=1,
        tensorboard_log="runs/chess_gpu",
        device=device,
    )

    # 5) Callbacks for logging & checkpoints
    csv_cb  = CSVLoggerCallback()
    ckpt_cb = CheckpointCallback(
        save_freq=10_000,
        save_path="models/",
        name_prefix="chess_ppo"
    )

    # 6) Train!
    model.learn(
        total_timesteps=10_000_000,
        callback=[csv_cb, ckpt_cb]
    )

    # 7) Final save
    model.save("models/chess_ppo_final")

if __name__ == "__main__":
    main()
