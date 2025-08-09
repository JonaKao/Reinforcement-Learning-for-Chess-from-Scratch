import time
import argparse
import numpy as np
import torch
import chess

# (Optional) same monkey-patch used in training; harmless at inference
import torch.distributions.categorical as _cat
_original_cat_init = _cat.Categorical.__init__
def _patched_cat_init(self, probs=None, logits=None, validate_args=None):
    return _original_cat_init(self, probs=probs, logits=logits, validate_args=False)
_cat.Categorical.__init__ = _patched_cat_init

from sb3_contrib.common.wrappers.action_masker import ActionMasker
from sb3_contrib import MaskablePPO

from Chess_Environment_Stockfish_Opponent import ChessEnv
from action_mapping import index_to_move, move_to_index  # uses your existing mapping

def mask_fn(env) -> np.ndarray:
    mask = np.zeros(env.action_space.n, dtype=bool)
    for mv in env.board.legal_moves:
        mask[move_to_index[mv.uci()]] = True
    return mask

def print_header(board, ply):
    print("\n" + "=" * 60)
    move_no = (ply // 2) + 1
    side = "White" if board.turn == chess.WHITE else "Black"
    print(f"Move {move_no} — {side} to play")
    print("-" * 60)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/chess_ppo_32900000_steps.zip")
    ap.add_argument("--sleep_s", type=float, default=0.6)
    ap.add_argument("--max_plies", type=int, default=400)
    ap.add_argument("--deterministic", action="store_true",
                    help="Greedy policy instead of sampling.")
    # Stockfish controls
    ap.add_argument("--engine_path", default=r"C:\Tools\Stockfish\stockfish.exe",
                    help="Full path to Stockfish binary or 'stockfish' if on PATH.")
    ap.add_argument("--elo", type=int, default=1000)
    ap.add_argument("--think_time", type=float, default=0.05,
                    help="Seconds per engine move; use tiny values to weaken.")
    ap.add_argument("--nodes", type=int, default=None,
                    help="Alternative to time limit; small (~50-200) keeps it weak.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # === Env vs Stockfish (no random openings) ===
    base_env = ChessEnv(
        engine_path=args.engine_path,
        target_elo=args.elo,
        think_time=args.think_time,
        nodes=args.nodes,
        randomize_start_color=True,
    )
    env = ActionMasker(base_env, mask_fn)

    print(f"[watch] Using engine path: {base_env.engine_path}")
    print(f"[watch] Engine active? {base_env.engine is not None}")

    print(f"Loading checkpoint: {args.model_path}")
    model = MaskablePPO.load(args.model_path, env=env, device=device)
    print("Checkpoint loaded.")

    obs, _ = env.reset()
    ply = 0

    print("\nInitial position (clean start, no random openings):")
    print(base_env.board)

    while True:
        if base_env.done or ply >= args.max_plies:
            break

        print_header(base_env.board, ply)

        # Mask + model action (sample by default, or greedy with --deterministic)
        mask = mask_fn(base_env)
        action, _ = model.predict(
            obs,
            deterministic=args.deterministic,
            action_masks=mask
        )

        # Validate action and obtain SAN for agent move
        uci = index_to_move[int(action)]
        move = chess.Move.from_uci(uci)
        if move not in base_env.board.legal_moves:
            print(f"Model proposed illegal move: {uci}. Ending game.")
            break

        # Keep a copy to format SAN strings
        board_before = base_env.board.copy()
        pre_stack_len = len(base_env.board.move_stack)
        agent_san = board_before.san(move)
        side_moved = "White" if board_before.turn == chess.WHITE else "Black"

        # Step the env (this also triggers Stockfish reply inside the env)
        obs, reward, done, truncated, info = env.step(int(action))

        # Determine if engine replied and print both moves
        post_stack_len = len(base_env.board.move_stack)
        print(f"{side_moved} (Agent) plays: {agent_san} ({uci})  | reward: {reward:+.3f}")

        # If the env made the engine move too, the move stack advanced by 2
        if post_stack_len - pre_stack_len == 2:
            agent_move, engine_move = base_env.board.move_stack[pre_stack_len:post_stack_len]
            # Get SAN for engine move using the intermediate position
            tmp = board_before.copy()
            tmp.push(agent_move)
            engine_san = tmp.san(engine_move)
            print(f"Stockfish replies: {engine_san} ({engine_move.uci()})")

        print(base_env.board)

        ply += 1
        time.sleep(args.sleep_s)
        if done:
            break

    print("\n" + "=" * 60)
    print("Game Over")
    outcome = base_env.board.outcome()
    try:
        print(f"Result: {base_env.board.result()}  |  Outcome: {outcome}")
    except Exception:
        print("Result: (could not determine)")

if __name__ == "__main__":
    main()
