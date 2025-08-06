import chess
import chess.engine
import gymnasium as gym
from gymnasium import spaces
from action_mapping import index_to_move
import numpy as np

class ChessEnv(gym.Env):
    def __init__(self):
        super(ChessEnv, self).__init__() # Initialize the environment
        self.board = chess.Board() # Initialize a new chess board
        self.action_space = spaces.Discrete(4672)  # All possible UCI moves (approx.)
        self.observation_space = spaces.Box(low=0, high=1, shape=(8, 8, 12), dtype=np.int8) # One-hot encoding (so sum is 1, thanks Prof. Koppe) for 12 piece types on an 8x8 board, def. the state input shape. Using a 3D tensor, 8 rows, 8 columns, and 12 planes for each piece type. Since one-hot encoding is used, the values are either 0 or 1 i.e. the piece is either present or not
        self.current_player = chess.WHITE  # White always starts
        self.done = False

    def reset(self, seed=None, options=None): # Reset the environment to the initial state
        super().reset(seed=seed)
        self.board.reset()
        self.done = False
        observation = self._get_obs()  # Get the initial observation
        self.current_player = chess.WHITE
        info = {}  # Additional info can be added here if needed
        return observation, info  # Return the initial observation and an empty info dict

    def _get_obs(self):
        # One-hot encode board into (8,8,12) planes
        if not self.current_player:
            board_planes = np.flip(board_planes, axis=(0, 1))  # Flip vertically and horizontally
        board_planes = np.zeros((8, 8, 12), dtype=np.int8)
        piece_to_plane = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                plane = piece_to_plane[piece.piece_type]
                if piece.color == chess.BLACK:
                    plane += 6  # Black pieces get planes 6–11
                row, col = divmod(square, 8)
                board_planes[row][col][plane] = 1
        return board_planes

    def step(self, action):
        uci_move = index_to_move.get(action)
        move = chess.Move.from_uci(uci_move)

        if move not in self.board.legal_moves:
            reward = -1
            self.done = True
            return self._get_obs(), reward, self.done, {}
        
        self.board.push(move) # Apply the move to the board
        reward = 0 # Default reward for a valid move
        if self.board.is_checkmate():
            reward = 1
            self.done = True # if the agent checkmates the opponent, end the episode with a win, so reward is 1
        elif self.board.is_stalemate() or self.board.is_insufficient_material(): # if the game ends in stalemate or insufficient material, end the episode with a weak penalty
            reward = -0.25
            self.done = True
        elif self.board.can_claim_fifty_moves(): # if the game can be claimed as a draw due to fifty-move rule, end the episode with a weak penalty
            reward = -0.5
            self.done = True

        return self._get_obs(), reward, self.done, {} #for now we return the new observation, reward, done flag, and an empty info dict (maybe I should add some info about the game state here later. idk yet)

    def render(self, mode='human'): # Render the current state of the board
        print(self.board)

    def legal_actions(self): # Return a list of legal actions (moves) available in the current state
        return list(self.board.legal_moves)
