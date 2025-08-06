import numpy as np
import torch
# Monkey-patch to disable Categorical validation for maskable distribution
import torch.distributions.categorical as _cat
_original_cat_init = _cat.Categorical.__init__
def _patched_cat_init(self, probs=None, logits=None, validate_args=None):
    return _original_cat_init(self, probs=probs, logits=logits, validate_args=False)
_cat.Categorical.__init__ = _patched_cat_init

from sb3_contrib.common.wrappers.action_masker import ActionMasker
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from callback_logging import CSVLoggerCallback
import os
import re

#from Chess_Environment import ChessEnv
from action_mapping    import move_to_index

# Build mask_fn for legal moves
def mask_fn(env) -> np.ndarray:
    mask = np.zeros(env.action_space.n, dtype=bool)
    for mv in env.board.legal_moves:
        mask[ move_to_index[mv.uci()] ] = True
    return mask


def main():
    # ensure models folder exists
    os.makedirs("models", exist_ok=True)

    # Choose device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Instantiate and wrap env
    base_env = ChessEnv()
    env      = ActionMasker(base_env, mask_fn)

    checkpoint_path = "models/chess_ppo_28600000_steps.zip"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        model = MaskablePPO.load(
            checkpoint_path,
            env=env,
            device=device,
        )
        print(f"Checkpoint loaded! Current timesteps: {model.num_timesteps}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}! Starting fresh...")
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            env,
            ent_coef=0.02, #wanted to promote more exploration
            verbose=1,
            tensorboard_log="runs/chess_gpu",
            device=device,
        )
        print("New model created! Current timesteps:", model.num_timesteps)

    # Callbacks for logging & checkpoints
    csv_cb  = CSVLoggerCallback()
    ckpt_cb = CheckpointCallback(
        save_freq=100_000,
        save_path="models/",
        name_prefix="chess_ppo"
    )
    print("Beginning/continuing training from", model.num_timesteps, "steps")
    # Train or continue training without resetting timestep counter
    model.learn(
        total_timesteps=model.num_timesteps + 3_000_000,
        reset_num_timesteps=False,
        callback=[csv_cb, ckpt_cb]
    )

    # Final save
    model.save(f"models/chess_ppo_final_{model.num_timesteps}_steps")
    print("Training complete! Final model saved as models/chess_ppo_final_{model.num_timesteps}_steps.zip")

if __name__ == "__main__":
    main()
