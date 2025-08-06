# self_play.py

import numpy as np

# Correctly import ActionMasker from the wrappers subpackage
from sb3_contrib.common.wrappers.action_masker import ActionMasker  # sb3-contrib v2.x :contentReference[oaicite:1]{index=1}

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from Chess_Environment import ChessEnv
from action_mapping import move_to_index
from callback_logging import CSVLoggerCallback

callback = CSVLoggerCallback("models/metrics.csv")

def mask_fn(env: ChessEnv) -> np.ndarray:
    """
    Given the current ChessEnv, return a boolean mask of shape (n_actions,)
    where True indicates a legal move.
    """
    mask = np.zeros(env.action_space.n, dtype=bool)
    for move in env.board.legal_moves:
        idx = move_to_index[move.uci()]
        mask[idx] = True
    return mask

def main():
    # 1) Instantiate your ChessEnv
    base_env = ChessEnv()

    # 2) Wrap it so the policy only samples legal moves
    env = ActionMasker(base_env, mask_fn)  # Wrap environment :contentReference[oaicite:2]{index=2}

    # 3) Build a MaskablePPO agent
    model = MaskablePPO(
        policy=MaskableActorCriticPolicy,
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
    )

    # 4) Train the agent
    model.learn(total_timesteps=500_000, callback=callback)

if __name__ == "__main__":
    main()
