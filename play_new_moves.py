# play_console_manual_mask.py

import time
import numpy as np
import torch
import random

from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO

from Chess_Environment import ChessEnv
from action_mapping    import move_to_index

def make_env():
    return ChessEnv()

def main():
    # 0) reseed so each run differs
    seed = int(time.time() * 1e6) % (2**32 - 1)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    # 1) wrap env so we can load it into the model
    vec_env = DummyVecEnv([make_env])

    # 2) load your trained PPO
    model = MaskablePPO.load(
        "models/chess_ppo_28600000_steps",
        env=vec_env,
        device="cuda"
    )

    # 3) grab the raw env for board & legal‐move info
    raw_env = vec_env.envs[0]

    # 4) reset
    obs = vec_env.reset()

    # 5) play 2–5 random opening plies to diversify positions
    n_open = np.random.randint(2, 6)
    for _ in range(n_open):
        mask = np.zeros(raw_env.action_space.n, dtype=bool)
        for mv in raw_env.board.legal_moves:
            mask[move_to_index[mv.uci()]] = True
        choice = np.random.choice(np.nonzero(mask)[0])
        obs, _, dones, _ = vec_env.step([int(choice)])
        if dones[0]:
            break  # very rare in random opening
    print(f"=== Did {n_open} random opening moves ===")

    # 6) render the resulting position
    print("=== Position after randomization ===")
    raw_env.render()
    time.sleep(1.0)

    # 7) now play the rest stochastically
    done = False
    while not done:
        # build legal‐move mask
        mask = np.zeros(raw_env.action_space.n, dtype=bool)
        for mv in raw_env.board.legal_moves:
            mask[ move_to_index[mv.uci()] ] = True

        # get policy probs
        obs_tensor = torch.as_tensor(obs, device=model.device)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        dist = model.policy.get_distribution(obs_tensor, action_masks=None)
        probs = dist.distribution.probs.detach().cpu().numpy().flatten()

        # zero out illegal
        legal_probs = probs * mask
        total = legal_probs.sum()

        # ε–greedy: 10% pure random over legal
        if np.random.rand() < 0.1 or total == 0:
            legal_idxs = np.nonzero(mask)[0]
            action = int(np.random.choice(legal_idxs))
        else:
            # sample from normalized policy
            prob_dist = legal_probs / total
            action = int(np.random.choice(len(prob_dist), p=prob_dist))

        # step & render
        obs, rewards, dones, infos = vec_env.step([action])
        done   = bool(dones[0])
        reward = rewards[0]

        raw_env.render()
        time.sleep(0.3)

    print(f"=== Game over, reward = {reward:.2f} ===")

if __name__ == "__main__":
    main()
