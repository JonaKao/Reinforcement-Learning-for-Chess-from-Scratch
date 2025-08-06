# play_console.py

import time
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO

from Chess_Environment import ChessEnv    # :contentReference[oaicite:2]{index=2}
from action_mapping    import move_to_index  # :contentReference[oaicite:3]{index=3}

def make_env():
    """Return a fresh ChessEnv for the VecEnv."""
    return ChessEnv()

def main():
    # 1) Wrap your raw ChessEnv in a DummyVecEnv
    vec_env = DummyVecEnv([make_env])

    # 2) Load your trained model (keep the VecEnv you just created)
    model = MaskablePPO.load(
        "models/chess_ppo_180000_steps",  # adjust path if needed
        env=vec_env,
        device="cpu",
    )

    # 3) Grab the inner env for rendering and masking
    raw_env = vec_env.envs[0]

    # 4) Reset (VecEnv.reset() => obs array)
    obs = vec_env.reset()
    done = False

    print("=== Starting position ===")
    raw_env.render()
    time.sleep(1.0)

    # 5) Play until done
    while not done:
        # Build a boolean mask of length n_actions
        mask = np.zeros(raw_env.action_space.n, dtype=bool)
        for mv in raw_env.board.legal_moves:
            idx = move_to_index[mv.uci()]
            mask[idx] = True

        # Ask the model for its best legal move
        # Note: action_masks must be a list (one per env), so we wrap in [ ... ]
        actions, _ = model.predict(
            obs,
            action_masks=[mask],
            deterministic=True
        )

        # Step the VecEnv (returns lists/arrays of length 1)
        obs, rewards, dones, infos = vec_env.step(actions)
        done   = bool(dones[0])
        reward = rewards[0]

        # Render the updated board
        raw_env.render()
        time.sleep(0.5)

    print(f"=== Game over, reward = {reward:.2f} ===")

if __name__ == "__main__":
    main()
