# watch_agent.py

import sys, os, re, glob, time
import numpy as np
import torch

from sb3_contrib.common.wrappers.action_masker import ActionMasker
from sb3_contrib import MaskablePPO

from Chess_Environment import ChessEnv
from action_mapping import move_to_index, index_to_move

def find_latest_checkpoint(models_dir="models"):
    """
    Scan `models_dir` for files named chess_ppo_<N>_steps.zip
    and return the one with the highest step count.
    """
    pattern = re.compile(r"chess_ppo_(\d+)_steps\.zip$")
    best_steps = -1
    best_path = None

    for path in glob.glob(os.path.join(models_dir, "*.zip")):
        m = pattern.search(os.path.basename(path))
        if m:
            steps = int(m.group(1))
            if steps > best_steps:
                best_steps = steps
                best_path = path

    if best_path is None:
        raise FileNotFoundError(f"No checkpoint chess_ppo_<N>_steps.zip found in {models_dir}")
    return best_path

def resolve_model_path(arg):
    """
    Given an argument like "white.zip" or "chess_ppo_100000_steps.zip",
    try (in order):
      1) exactly as given in cwd
      2) inside ./models/
      3) adding/removing .zip suffix in ./models/
    """
    # 1) as given
    if os.path.isfile(arg):
        return arg
    # 2) in models/
    m2 = os.path.join("models", os.path.basename(arg))
    if os.path.isfile(m2):
        return m2
    # 3) ensure .zip suffix
    base = os.path.splitext(arg)[0]
    m3 = os.path.join("models", base + ".zip")
    if os.path.isfile(m3):
        return m3
    raise FileNotFoundError(f"Could not find model file: {arg}")

def mask_fn(env) -> np.ndarray:
    """
    Unwraps the Maskable wrapper then masks illegal moves.
    """
    real_env = env.env
    mask = np.zeros(real_env.action_space.n, dtype=bool)
    for move in real_env.board.legal_moves:
        mask[move_to_index[move.uci()]] = True
    return mask

def main():
    # ---------------------
    # Arg parsing
    # ---------------------
    args = sys.argv[1:]
    if len(args) == 0:
        latest = find_latest_checkpoint()
        white_path = black_path = latest
        print(f"No args: using latest checkpoint → {latest}")
    elif len(args) == 1:
        white_path = black_path = args[0]
        print(f"One arg: using {args[0]} for both sides")
    elif len(args) == 2:
        white_path, black_path = args
    else:
        print("Usage:")
        print("  python watch_agent.py             # latest vs latest")
        print("  python watch_agent.py model.zip   # same model both sides")
        print("  python watch_agent.py w.zip b.zip # white vs black")
        sys.exit(1)

    # ---------------------
    # Resolve actual file locations
    # ---------------------
    try:
        white_file = resolve_model_path(white_path)
        black_file = resolve_model_path(black_path)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    # ---------------------
    # Device & load
    # ---------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nWhite agent → {white_file} on {device}")
    print(f"Black agent → {black_file} on {device}\n")

    # Create & mask environment
    base_env = ChessEnv()  
    env = ActionMasker(base_env, mask_fn)

    # SB3.load will append .zip if needed
    white_id = os.path.splitext(white_file)[0]
    black_id = os.path.splitext(black_file)[0]

    model_white = MaskablePPO.load(white_id, env=env, device=device)
    model_black = MaskablePPO.load(black_id, env=env, device=device)

    # ---------------------
    # Play loop
    # ---------------------
    obs, _ = env.reset()
    real_env = env.env
    done = False
    turn = 0  # even=white, odd=black

    print("Starting game:")
    print(real_env.board, "\n")

    while not done:
        if turn % 2 == 0:
            action, _ = model_white.predict(obs, deterministic=True)
            player = "White"
        else:
            action, _ = model_black.predict(obs, deterministic=True)
            player = "Black"

        obs, reward, terminated, truncated, info = env.step(action)
        real_env = env.env

        print(f"{player} plays {index_to_move[int(action)]}")
        print(real_env.board, "\n")
        time.sleep(0.5)

        done = terminated or truncated
        turn += 1

    print(f"=== Game Over: {info.get('result','unknown')} ===")

if __name__ == "__main__":
    main()
