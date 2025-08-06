# action_mapping.py

import chess

uci_moves = []

for from_sq in chess.SQUARES:
    for to_sq in chess.SQUARES:
        # skip no-op
        if from_sq == to_sq:
            continue

        # plain move
        uci_moves.append(
            chess.square_name(from_sq)
            + chess.square_name(to_sq)
        )

        # white promotions (rank 7 → 8)
        if chess.square_rank(from_sq) == 6 and chess.square_rank(to_sq) == 7:
            for promo in ['q', 'r', 'b', 'n']:
                uci_moves.append(
                    chess.square_name(from_sq)
                    + chess.square_name(to_sq)
                    + promo
                )

        # black promotions (rank 2 → 1)
        if chess.square_rank(from_sq) == 1 and chess.square_rank(to_sq) == 0:
            for promo in ['q', 'r', 'b', 'n']:
                uci_moves.append(
                    chess.square_name(from_sq)
                    + chess.square_name(to_sq)
                    + promo
                )

# remove duplicates & sort
uci_moves = sorted(set(uci_moves))

# build maps
move_to_index = {move: idx for idx, move in enumerate(uci_moves)}
index_to_move = {idx: move for move, idx in move_to_index.items()}
