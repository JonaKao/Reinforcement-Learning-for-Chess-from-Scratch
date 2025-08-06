import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers.action_masker import ActionMasker
from Chess_Environment import ChessEnv
from action_mapping import move_to_index
from action_mapping import index_to_move
import numpy as np

def mask_fn(env):
    mask = np.zeros(env.action_space.n, dtype=bool)
    for mv in env.board.legal_moves:
        mask[ move_to_index[mv.uci()] ] = True
    return mask

def main():
    # Load model and env
    base_env = ChessEnv()
    env      = ActionMasker(base_env, mask_fn)
    model = MaskablePPO.load("models/chess_ppo_final", env=env)

    num_episodes = 20
    for episode in range(num_episodes):
        obs, _ = env.reset()
        legal_mask = mask_fn(base_env)
        legal_indices = np.where(legal_mask)[0]
        print(f"Legal move indices: {legal_indices}")
        print(f"Legal UCIs: {[index_to_move[i] for i in legal_indices]}")
        print(f"Episode {episode+1}: {base_env.starting_color} moves first")
        done = False
        move_list = []
        while not done:
            mask = mask_fn(base_env)
            action, _ = model.predict(obs, action_masks=mask)
            obs, reward, done, _, _ = env.step(action)
            move_uci = base_env.board.peek().uci() if base_env.board.move_stack else "None"
            print(f"  Action index: {action}, UCI: {move_uci}, Reward: {reward}, Done: {done}")
            move_list.append(move_uci)

        # Log winner/result
        result = "draw"
        if base_env.board.is_checkmate():
            winner = "white" if not base_env.board.turn else "black"
            result = f"{winner} wins by checkmate"
        elif base_env.board.is_stalemate():
            result = "stalemate"

        print(f"Starter: {base_env.starting_color}, Result: {result}, Moves: {move_list}")

if __name__ == "__main__":
    main()
