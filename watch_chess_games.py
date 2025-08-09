import time
import numpy as np
import torch
import chess

# (Optional) same monkey-patch we use in training; harmless at inference and avoids mask-validation quirks
import torch.distributions.categorical as _cat
_original_cat_init = _cat.Categorical.__init__
def _patched_cat_init(self, probs=None, logits=None, validate_args=None):
    return _original_cat_init(self, probs=probs, logits=logits, validate_args=False)
_cat.Categorical.__init__ = _patched_cat_init

from sb3_contrib.common.wrappers.action_masker import ActionMasker
from sb3_contrib import MaskablePPO

from Chess_Environment import ChessEnv       # uses random start + 8–10 random opening plies
from action_mapping import index_to_move, move_to_index

def mask_fn(env) -> np.ndarray:
    mask = np.zeros(env.action_space.n, dtype=bool)
    for mv in env.board.legal_moves:
        mask[move_to_index[mv.uci()]] = True
    return mask

def print_header(board, ply):
    print("\n" + "="*60)
    move_no = (ply // 2) + 1
    side = "White" if board.turn == chess.WHITE else "Black"
    print(f"Move {move_no} — {side} to play")
    print("-"*60)

def main(
    model_path="models/chess_ppo_28600000_steps.zip",
    sleep_s=0.5,
    max_plies=400,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Use training env behavior: random start + random opening plies
    base_env = ChessEnv()
    env = ActionMasker(base_env, mask_fn)

    print(f"Loading checkpoint: {model_path}")
    model = MaskablePPO.load(model_path, env=env, device=device)
    print("Checkpoint loaded.")

    obs, _ = env.reset()
    ply = 0

    print("\nInitial position (after env.reset() which may include random opening plies):")
    print(base_env.board)  # ASCII board

    while True:
        if base_env.done or ply >= max_plies:
            break

        print_header(base_env.board, ply)

        mask = mask_fn(base_env)
        # Sample from the policy, like training-time behavior (not greedy)
        action, _ = model.predict(obs, deterministic=False, action_masks=mask)

        uci = index_to_move[int(action)]
        move = chess.Move.from_uci(uci)
        if move not in base_env.board.legal_moves:
            print(f"Model proposed illegal move: {uci}. Ending game.")
            break

        board_copy = base_env.board.copy()
        san = board_copy.san(move)

        obs, reward, done, truncated, info = env.step(int(action))

        side_moved = "White" if board_copy.turn == chess.WHITE else "Black"
        print(f"{side_moved} plays: {san} ({uci})  | reward: {reward:+.3f}")
        print(base_env.board)

        ply += 1
        time.sleep(sleep_s)
        if done:
            break

    print("\n" + "="*60)
    print("Game Over")
    outcome = base_env.board.outcome()
    try:
        print(f"Result: {base_env.board.result()}  |  Outcome: {outcome}")
    except Exception:
        print("Result: (could not determine)")

if __name__ == "__main__":
    main(sleep_s=0.6)
