# play_console_manual_mask.py

import time
import numpy as np
import torch

from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO

from Chess_Environment import ChessEnv      # my env
from action_mapping    import move_to_index # my mapping

def make_env():
    return ChessEnv()

def main():
    # 1) wrap env so we can load it into the model
    vec_env = DummyVecEnv([make_env])

    # 2) load trained PPO (no mask argument!)
    model = MaskablePPO.load("models/chess_ppo_28600000_steps", env=vec_env, device="cpu")

    # 3) grab the raw env for board & legal‐move info
    raw_env = vec_env.envs[0]

    # 4) reset
    obs = vec_env.reset()
    done = False

    print("=== Starting position ===")
    raw_env.render()
    time.sleep(1.0)

    while not done:
        # — build the legal‐move mask —
        mask = np.zeros(raw_env.action_space.n, dtype=bool)
        for mv in raw_env.board.legal_moves:
            mask[ move_to_index[mv.uci()] ] = True

        # — get the network's unmasked dist —
        obs_tensor = torch.as_tensor(obs, device=model.device)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        dist = model.policy.get_distribution(obs_tensor, action_masks=None)
        probs = dist.distribution.probs.detach().cpu().numpy().flatten()

        # — zero out illegal moves & **stochastic sample** —
        legal_probs = probs * mask
        total = legal_probs.sum()
        if total > 0:
            prob_dist = legal_probs / total
            action = int(np.random.choice(len(prob_dist), p=prob_dist))
        else:
            # fallback: pick a random legal move
            legal_idxs = np.nonzero(mask)[0]
            action = int(np.random.choice(legal_idxs))

        # — step & render —
        obs, rewards, dones, infos = vec_env.step([action])
        done   = bool(dones[0])
        reward = rewards[0]

        raw_env.render()
        time.sleep(0.5)

    print(f"=== Game over, reward = {reward:.2f} ===")

if __name__ == "__main__":
    main()
