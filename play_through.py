# play_through.py

import numpy as np
import torch
from sb3_contrib.common.wrappers.action_masker import ActionMasker
from sb3_contrib import MaskablePPO

from Chess_Environment import ChessEnv
from action_mapping import move_to_index, index_to_move

def mask_fn(env):
    mask = np.zeros(env.action_space.n, dtype=bool)
    for move in env.board.legal_moves:
        idx = move_to_index[move.uci()]
        mask[idx] = True
    return mask

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = ActionMasker(ChessEnv(), mask_fn)

    # Load your best checkpoint; adjust the path as needed
    model = MaskablePPO.load(
        "./models/chess_ppo_1000000_steps.zip",
        env=env,
        device=device
    )

    obs, _ = env.reset()
    done = False
    print(env.envs[0].board)  # initial board

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        print("\nMove:", index_to_move[int(action)])
        print(env.envs[0].board)
        done = terminated or truncated

    print("Game over, reward:", reward)

if __name__ == "__main__":
    main()
