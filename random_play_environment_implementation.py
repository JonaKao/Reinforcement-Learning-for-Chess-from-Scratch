import random
from Chess_Environment import ChessEnv
from action_mapping import move_to_index

class RandomOpponentChessEnv(ChessEnv):
    def __init__(self, rl_plays_white=True):
        super().__init__()
        self.rl_plays_white = rl_plays_white

    def step(self, action):
        # Decide who the RL agent is (white or black)
        agent_color = chess.WHITE if self.rl_plays_white else chess.BLACK

        # Only allow RL agent to move on their color
        if self.board.turn != agent_color:
            # Random opponent's move
            legal_moves = list(self.board.legal_moves)
            if legal_moves:
                move = random.choice(legal_moves)
                action_idx = move_to_index[move.uci()]
                obs, reward, done, truncated, info = super().step(action_idx)
                if done:
                    return obs, reward, done, truncated, info
            # Now it's RL agent's turn

        # RL agent's move
        obs, reward, done, truncated, info = super().step(action)
        return obs, reward, done, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # At the start of every episode, optionally randomize which color is RL
        self.rl_plays_white = random.choice([True, False])  # 50% chance each
        return obs, info
