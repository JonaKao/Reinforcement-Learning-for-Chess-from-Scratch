# Chess_Environment.py

import chess
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from action_mapping import index_to_move
import random


#going to build intermittent rewards for captures to encourage the agent to capture pieces, makes draws less likely
PIECE_VALUES = {
    chess.PAWN:   0.01,
    chess.KNIGHT: 0.03,
    chess.BISHOP: 0.03,
    chess.ROOK:   0.05,
    chess.QUEEN:  0.09,
}

class ChessEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.board = chess.Board()
        self.action_space = spaces.Discrete(len(index_to_move))
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(8, 8, 12),
            dtype=np.int8
        )
        self.done = False
    @property
    def action_mask(self):
        mask = np.zeros(self.action_space.n, dtype=bool)
        for mv in self.board.legal_moves:
            mask[ move_to_index[mv.uci()] ] = True
        return mask

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.board.reset()
        if np.random.rand() < 0.5:
            self.board.turn = chess.BLACK  # Randomly choose who starts
            self.starting_color = "black"
        else:
            self.board.turn = chess.WHITE
            self.starting_color = "white"
        self.done = False

        num_opening_moves = random.randint(8, 10)  # e.g., 2 to 6 random moves (that's what it was intially, after another episode of consistent wins at low move counts (circa 20M mark), I decided to increase it)
        for _ in range(num_opening_moves):
            legal_moves = list(self.board.legal_moves)
            if not legal_moves or self.board.is_game_over():
                break  # Don't continue if game ends in the random sequence!
            move = random.choice(legal_moves)
            self.board.push(move)

        print(f"[ChessEnv.reset()] Starting color: {self.starting_color}")
        return self._get_obs(), {}

    def _get_obs(self):
        planes = np.zeros((8, 8, 12), dtype=np.int8)
        piece_plane = {
            chess.PAWN:   0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK:   3,
            chess.QUEEN:  4,
            chess.KING:   5,
        }
        for sq in chess.SQUARES:
            p = self.board.piece_at(sq)
            if p:
                plane = piece_plane[p.piece_type] + (6 if p.color == chess.BLACK else 0)
                r, c = divmod(sq, 8)
                planes[r, c, plane] = 1
        return planes

    def step(self, action):
        # 1) unwrap numpy arrays
        if isinstance(action, np.ndarray):
            action = int(action)

        # 2) guard out-of-range
        if action not in index_to_move:
            raise ValueError(
                f"Action {action} invalid; must be in 0..{len(index_to_move)-1}"
            )

        uci = index_to_move[action]

        # 3) catch any malformed UCI
        try:
           move = chess.Move.from_uci(uci)
        except chess.InvalidMoveError:
            self.done = True
            return self._get_obs(), -1, True, False, {}

        # 4) illegal-move penalty
        if move not in self.board.legal_moves:
            self.done = True
            return self._get_obs(), -1, True, False, {}

        captured_piece = self.board.piece_at(move.to_square)

        # 5) apply
        self.board.push(move)
        reward =- 0.002  # Small penalty for each move to encourage faster play
        if captured_piece:
            # PIECE_VALUES defined at top of file
            reward += PIECE_VALUES.get(captured_piece.piece_type, 0.0)
        # 6) terminal
        if self.board.is_checkmate():
            reward += 2.5
            self.done = True
        elif (
            self.board.is_stalemate()
            or self.board.is_insufficient_material()
            or self.board.can_claim_fifty_moves()
            or self.board.can_claim_threefold_repetition()
        ):
            reward += -1
            self.done = True

        # 7) return exactly five values
        return self._get_obs(), reward, self.done, False, {}

    def render(self, mode='human'):
        print(self.board)

    def legal_actions(self):
        return list(self.board.legal_moves)
