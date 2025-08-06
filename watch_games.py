# watch_games.py

import os
import re
import time
import random
import glob
import numpy as np
import torch
import chess
from sb3_contrib.common.wrappers.action_masker import ActionMasker
from sb3_contrib import MaskablePPO

from Chess_Environment import ChessEnv
from action_mapping import move_to_index, index_to_move

def find_latest_checkpoint(models_dir="models"):
    pattern = re.compile(r"chess_ppo_(\d+)_steps\.zip$")
    candidates = glob.glob(os.path.join(models_dir, "chess_ppo_*_steps.zip"))
    best = None
    best_steps = -1
    for path in candidates:
        m = pattern.search(os.path.basename(path))
        if m:
            steps = int(m.group(1))
            if steps > best_steps:
                best_steps = steps
                best = path
    if best is None:
        raise FileNotFoundError(f"No checkpoint matching chess_ppo_<N>_steps.zip in {models_dir}")
    return best, best_steps

def mask_fn(env) -> np.ndarray:
    # env here is the ActionMasker wrapper around ChessEnv
    real_env = env.env
    mask = np.zeros(real_env.action_space.n, dtype=bool)
    for move in real_env.board.legal_moves:
        mask[move_to_index[move.uci()]] = True
    return mask

def play_one_game(model, env):
    obs, _ = env.reset()
    real_env = env.env
    done = False

    print("\n=== New Game ===")
    print(real_env.board)

    while not done:
        board = real_env.board
        if board.turn == chess.WHITE:
            action, _ = model.predict(obs, deterministic=True)
            print(f"\nAgent (White) plays: {index_to_move[int(action)]}")
        else:
            opp_move = random.choice(list(board.legal_moves))
            action = move_to_index[opp_move.uci()]
            print(f"\nOpponent (Black) plays: {opp_move.uci()}")

        obs, reward, terminated, truncated, info = env.step(action)
        real_env = env.env
        print(real_env.board)
        time.sleep(0.5)
        done = terminated or truncated

    fb = real_env.board
    if fb.is_checkmate():
        winner = "White" if fb.turn == chess.BLACK else "Black"
        result = "1-0" if winner == "White" else "0-1"
    elif fb.is_stalemate() or fb.is_insufficient_material() or fb.can_claim_fifty_moves():
        result = "1/2-1/2"
    else:
        result = "*"

    print(f"\n=== Game over: {result} ===")
    return result

def main(num_games=5):
    ckpt_path, steps = find_latest_checkpoint("models")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading agent from {ckpt_path} ({steps} steps) onto {device}…")

    base_env = ChessEnv()
    env = ActionMasker(base_env, mask_fn)

    # strip ".zip" for SB3 load
    model = MaskablePPO.load(ckpt_path[:-4], env=env, device=device)

    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "*": 0}
    for i in range(num_games):
        print(f"\n>>> Playing game {i+1}/{num_games}")
        res = play_one_game(model, env)
        results[res] += 1

    print("\nSummary over", num_games, "games:\n", results)

if __name__ == "__main__":
    main()
